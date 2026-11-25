package parser

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWePMReleaseFiles(t *testing.T) {
	dir := t.TempDir()
	files := map[string]map[string]string{}
	for _, name := range []string{"language-BF16.gguf", "language-Q4_K_M.gguf", "multimodal-BF16.gguf"} {
		data := []byte(name)
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
		files[name] = map[string]string{"sha256": fmt.Sprintf("%x", sha256.Sum256(data))}
	}
	body, err := json.Marshal(map[string]any{"format": "wepm-gguf-release-v1", "architecture": "task_slots_v1", "files": files})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bundle.json"), body, 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ source, lang string }{{dir, "language-Q4_K_M.gguf"}, {filepath.Join(dir, "language-BF16.gguf"), "language-BF16.gguf"}} {
		got, err := fileDigestMap(tc.source)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 2 {
			t.Fatalf("selected %d files", len(got))
		}
		for _, name := range []string{tc.lang, "multimodal-BF16.gguf"} {
			if got[filepath.Join(dir, name)] != "sha256:"+files[name]["sha256"] {
				t.Fatalf("missing or wrong digest for %s", name)
			}
		}
	}
	delete(files, "language-Q4_K_M.gguf")
	bf16Only, err := json.Marshal(map[string]any{"format": "wepm-gguf-release-v1", "architecture": "task_slots_v1", "files": files})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bundle.json"), bf16Only, 0600); err != nil {
		t.Fatal(err)
	}
	got, err := fileDigestMap(dir)
	if err != nil || len(got) != 2 || got[filepath.Join(dir, "language-BF16.gguf")] == "" {
		t.Fatalf("BF16-only release: %v, %v", got, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "multimodal-BF16.gguf"), []byte("truncated"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := fileDigestMap(dir); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("corrupted bundle: %v", err)
	}
}
