package server

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
)

type wepmTensor struct {
	source   gguf.TensorInfo
	name     string
	dims     []uint64
	temporal int
	vision   bool
}

var wepmVisionPrefixes = map[string]string{
	"merger.linear_fc1.": "mm.0.",
	"merger.linear_fc2.": "mm.2.",
	"merger.norm.":       "v.post_ln.",
	"patch_embed.proj.":  "v.patch_embd.",
	"pos_embed.":         "v.position_embd.",
}

var wepmBlockNames = map[string]string{
	"attn.proj":      "attn_out",
	"attn.qkv":       "attn_qkv",
	"mlp.linear_fc1": "ffn_up",
	"mlp.linear_fc2": "ffn_down",
	"norm1":          "ln1",
	"norm2":          "ln2",
}

func wepmVisionName(source string) (string, error) {
	name := strings.TrimPrefix(source, "model.visual.")
	for prefix, replacement := range wepmVisionPrefixes {
		if suffix, ok := strings.CutPrefix(name, prefix); ok {
			return replacement + suffix, nil
		}
	}
	parts := strings.Split(name, ".")
	if len(parts) >= 4 && parts[0] == "blocks" {
		block, err := strconv.Atoi(parts[1])
		suffix := parts[len(parts)-1]
		stem, ok := wepmBlockNames[strings.Join(parts[2:len(parts)-1], ".")]
		if err == nil && block >= 0 && block < 24 && ok && (suffix == "weight" || suffix == "bias") {
			return fmt.Sprintf("v.blk.%d.%s.%s", block, stem, suffix), nil
		}
	}
	return "", fmt.Errorf("unsupported WePM vision tensor %q", source)
}

func validateWePMHeads(src *gguf.File, width uint64) error {
	want := map[string][]uint64{
		"embedding_head.weight": {width, 2048}, "decision_head.output.weight": {width, 256},
		"slot_pooler.embedding_slot": {width}, "slot_pooler.decision_slot": {width},
		"mgvb.input_norm.weight": {width}, "mgvb.input_proj.weight": {width, 512},
		"mgvb.query_proj.weight": {width * 2, 512}, "mgvb.null_query": {512}, "mgvb.global_queries": {512, 8},
		"mgvb.global_query_condition.weight": {512, 512}, "mgvb.global_key.weight": {512, 512}, "mgvb.global_value.weight": {512, 512},
		"mgvb.granularity_embedding.weight": {width, 3}, "mgvb.output_position.weight": {4, width},
	}
	for _, bank := range []string{"global", "region", "fine"} {
		prefix := "mgvb." + bank + "_residual."
		for name, shape := range map[string][]uint64{"norm.weight": {512}, "norm.bias": {512}, "fc1.weight": {512, 1024}, "fc2.weight": {1024, width}} {
			want[prefix+name] = shape
		}
	}
	for _, bank := range []string{"region", "fine"} {
		for name, shape := range map[string][]uint64{"query.weight": {512, 512}, "value.weight": {512, 512}, "position.weight": {4, 512}, "score.weight": {512, 1}} {
			want["mgvb."+bank+"_"+name] = shape
		}
	}
	for name, w := range want {
		t := src.TensorInfo("unimodel." + name)
		if t.Type != gguf.TensorTypeBF16 || !slices.Equal(t.Shape, w) {
			return fmt.Errorf("invalid WePM tensor %s: expected BF16 %v", name, w)
		}
	}
	return nil
}

// importWePMProjector packages the release vision tower and BF16 task heads for mtmd.
func importWePMProjector(path string) (manifest.Layer, error) {
	var empty manifest.Layer
	src, err := gguf.Open(path)
	if err != nil {
		return empty, err
	}
	defer src.Close()
	width := src.TensorInfo("unimodel.embedding_head.weight").Shape
	if len(width) != 2 || width[1] != 2048 || (width[0] != 2048 && width[0] != 2560) {
		return empty, fmt.Errorf("invalid WePM embedding head")
	}
	if err := validateWePMHeads(src, width[0]); err != nil {
		return empty, err
	}
	var tensors []wepmTensor
	for _, ti := range src.TensorInfos() {
		if ti.Type != gguf.TensorTypeBF16 || ti.NumBytes() <= 0 || ti.NumBytes() > 128<<20 {
			return empty, fmt.Errorf("invalid WePM BF16 tensor %s", ti.Name)
		}
		t := wepmTensor{source: ti, name: ti.Name, dims: ti.Shape, temporal: -1}
		if strings.HasPrefix(t.name, "model.visual.") {
			t.vision = true
			t.name, err = wepmVisionName(t.name)
			if err != nil {
				return empty, err
			}
			if t.name == "v.patch_embd.weight" {
				if !slices.Equal(ti.Shape, []uint64{1536, 1024}) {
					return empty, fmt.Errorf("invalid WePM temporal convolution")
				}
				t.dims = []uint64{16, 16, 3, 1024}
				t.temporal = 0
				tensors = append(tensors, t)
				t.name += ".1"
				t.temporal = 1
			}
		} else if !strings.HasPrefix(t.name, "unimodel.") {
			return empty, fmt.Errorf("unexpected WePM tensor %s", t.name)
		}
		tensors = append(tensors, t)
	}
	if err := src.Err(); err != nil {
		return empty, err
	}
	var meta bytes.Buffer
	var nk uint64
	put := func(b *bytes.Buffer, x any) {
		if e := binary.Write(b, binary.LittleEndian, x); e != nil {
			panic(e)
		}
	}
	str := func(b *bytes.Buffer, s string) {
		put(b, uint64(len(s)))
		b.WriteString(s)
	}
	kv := func(k string, kind uint32, v any) {
		nk++
		str(&meta, k)
		put(&meta, kind)
		switch x := v.(type) {
		case string:
			str(&meta, x)
		case []float32:
			put(&meta, uint32(6))
			put(&meta, uint64(len(x)))
			put(&meta, x)
		default:
			put(&meta, x)
		}
	}
	kv("general.architecture", 8, "clip")
	kv("general.type", 8, "mmproj")
	kv("clip.projector_type", 8, "qwen3vl_merger")
	kv("clip.has_vision_encoder", 7, true)
	kv("clip.use_gelu", 7, true)
	kv("clip.vision.wepm_task_slots_v1", 7, true)
	kv("clip.wepm.task_heads", 7, true)
	for _, v := range []struct {
		k string
		n uint32
	}{
		{"projection_dim", uint32(width[0])},
		{"image_size", 768},
		{"patch_size", 16},
		{"embedding_length", 1024},
		{"feed_forward_length", 4096},
		{"block_count", 24},
		{"attention.head_count", 16},
		{"spatial_merge_size", 2},
	} {
		kv("clip.vision."+v.k, 4, v.n)
	}
	kv("clip.vision.image_mean", 9, []float32{.5, .5, .5})
	kv("clip.vision.image_std", 9, []float32{.5, .5, .5})
	kv("clip.vision.attention.layer_norm_epsilon", 6, float32(1e-6))
	var header bytes.Buffer
	header.WriteString("GGUF")
	put(&header, uint32(3))
	put(&header, uint64(len(tensors)))
	put(&header, nk)
	header.Write(meta.Bytes())
	var offset int64
	for _, t := range tensors {
		str(&header, t.name)
		put(&header, uint32(len(t.dims)))
		put(&header, t.dims)
		typ := t.source.Type
		n := t.source.NumBytes()
		if t.vision {
			typ = gguf.TensorTypeF32
			n *= 2
		}
		if t.temporal >= 0 {
			n /= 2
		}
		put(&header, typ)
		put(&header, uint64(offset))
		offset += (n + 31) &^ 31
	}
	header.Write(make([]byte, (-header.Len())&31))
	out, err := os.CreateTemp(filepath.Dir(path), ".wepm-projector-*")
	if err != nil {
		return empty, err
	}
	defer os.Remove(out.Name())
	defer out.Close()
	if _, err = out.Write(header.Bytes()); err != nil {
		return empty, err
	}
	buf := make([]byte, 3*2*256*2*32)
	expanded := make([]byte, len(buf)*2)
	for _, t := range tensors {
		_, r, e := src.TensorReader(t.source.Name)
		if e != nil {
			return empty, e
		}
		written := int64(0)
		if !t.vision {
			written, e = io.Copy(out, r)
		} else {
			for left := t.source.NumBytes(); left > 0; {
				n := int(min(left, int64(len(buf))))
				if _, e = io.ReadFull(r, buf[:n]); e != nil {
					return empty, e
				}
				left -= int64(n)
				j := 0
				for i := 0; i < n; i += 2 {
					if t.temporal >= 0 && (i/(256*2))%2 != t.temporal {
						continue
					}
					expanded[j+2] = buf[i]
					expanded[j+3] = buf[i+1]
					j += 4
				}
				if _, e = out.Write(expanded[:j]); e != nil {
					return empty, e
				}
				written += int64(j)
			}
		}
		if e != nil {
			return empty, e
		}
		if _, e = out.Write(make([]byte, (-written)&31)); e != nil {
			return empty, e
		}
	}
	if err = out.Close(); err != nil {
		return empty, err
	}
	return manifest.NewLayerFromFile(out.Name(), manifest.MediaTypeImageProjector)
}
