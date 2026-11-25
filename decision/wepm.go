package decision

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// WePM scores one question with a direct head. Candidate indices follow the
// wire order, so the original JSON is forwarded to the runner after validation.
func validateWePMRequest(req Request) error {
	q, ok := req.Questions.Get("q")
	if !ok || req.Questions.Len() != 1 {
		return errors.New("WePM expects exactly questions.q")
	}
	state := bytes.TrimSpace(req.State)
	if len(state) == 0 || (state[0] != '"' && state[0] != '{' && !bytes.Equal(state, []byte("null"))) {
		return errors.New("WePM state must be an object, string, or null")
	}
	if len(req.Images) > 8 {
		return errors.New("WePM accepts at most eight images")
	}
	if err := validateImages(req.Images); err != nil {
		return err
	}
	var instruction string
	if err := json.Unmarshal(q.Instructions, &instruction); err != nil || strings.TrimSpace(instruction) == "" {
		return errors.New("WePM instructions must be a nonempty string")
	}
	var descriptions []string
	switch q.Type {
	case "noul":
		if q.Criteria != nil {
			return errors.New("WePM noul questions cannot include criteria")
		}
		return nil
	case "choice":
		var criteria map[string]string
		if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
			return errors.New("WePM choice criteria must be an object of descriptions")
		}
		for id, description := range criteria {
			if id == "" {
				return errors.New("WePM choice IDs must be nonempty")
			}
			descriptions = append(descriptions, description)
		}
	case "score":
		if err := json.Unmarshal(q.Criteria, &descriptions); err != nil {
			return errors.New("WePM score criteria must be an ordered list of descriptions")
		}
	default:
		return fmt.Errorf("unsupported WePM question type %q", q.Type)
	}
	if len(descriptions) < 2 || len(descriptions) > 255 {
		return errors.New("WePM requires 2–255 candidates")
	}
	for _, description := range descriptions {
		if strings.TrimSpace(description) == "" {
			return errors.New("WePM candidate descriptions must be nonempty strings")
		}
	}
	return nil
}
