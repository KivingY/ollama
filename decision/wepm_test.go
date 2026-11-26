package decision

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestWePMRequestLimits(t *testing.T) {
	for _, tc := range []struct {
		name, state, questions string
		valid                  bool
	}{
		{"choice order", `{"b":2,"a":1}`, `{"q":{"type":"choice","instructions":"pick","criteria":{"z":"last","a":"first"}}}`, true},
		{"null state", "null", `{"q":{"type":"noul","instructions":"visible?"}}`, true},
		{"empty state", `""`, `{"q":{"type":"noul","instructions":"visible?"}}`, true},
		{"absent state", "", `{"q":{"type":"noul","instructions":"visible?"}}`, false},
		{"array state", "[]", `{"q":{"type":"noul","instructions":"visible?"}}`, false},
		{"named question", `"image"`, `{"is_greeting":{"type":"noul","instructions":"test"}}`, true},
		{"two questions", `"image"`, `{"a":{"type":"noul","instructions":"test"},"b":{"type":"noul","instructions":"test"}}`, false},
		{"no questions", `"image"`, `{}`, false},
		{"implicit choice descriptions", `"image"`, `{"pick":{"type":"choice","instructions":"test","criteria":{"yes":null,"no":null}}}`, true},
		{"empty choice description", `"image"`, `{"pick":{"type":"choice","instructions":"test","criteria":{"yes":"","no":null}}}`, false},
		{"empty description", `"image"`, `{"q":{"type":"score","instructions":"rate","criteria":["","x"]}}`, false},
		{"empty instruction", `"image"`, `{"q":{"type":"noul","instructions":" "}}`, false},
		{"single option", `"image"`, `{"q":{"type":"choice","instructions":"pick","criteria":{"x":"x"}}}`, false},
		{"noul criteria", `"image"`, `{"q":{"type":"noul","instructions":"visible?","criteria":null}}`, false},
		{"255 candidates", `"image"`, `{"q":{"type":"score","instructions":"rate","criteria":[` + strings.Repeat(`"x",`, 254) + `"x"]}}`, true},
		{"256 candidates", `"image"`, `{"q":{"type":"score","instructions":"rate","criteria":[` + strings.Repeat(`"x",`, 255) + `"x"]}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := Request{State: json.RawMessage(tc.state)}
			if err := json.Unmarshal([]byte(tc.questions), &req.Questions); err != nil {
				t.Fatal(err)
			}
			if err := ValidateLimits(req, "wepm"); (err == nil) != tc.valid {
				t.Fatalf("ValidateLimits() = %v, want valid=%t", err, tc.valid)
			}
		})
	}
}
