package common

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ClaudeMessageAccumulator groups consecutive Claude messages by role.
type ClaudeMessageAccumulator struct {
	messages     [][]byte
	role         string
	content      [][]byte
	toolUseParts [][]byte
}

// NewClaudeMessageAccumulator creates an accumulator sized for the expected messages.
func NewClaudeMessageAccumulator(capacity int) *ClaudeMessageAccumulator {
	return &ClaudeMessageAccumulator{
		messages: NewRawArrayItems(int64(capacity)),
	}
}

// Append adds one Claude-shaped message to the current role turn.
func (a *ClaudeMessageAccumulator) Append(message []byte) {
	if len(message) == 0 {
		return
	}
	root := gjson.ParseBytes(message)
	role := root.Get("role").String()
	if role != "user" && role != "assistant" {
		return
	}
	parts := claudeMessageContentParts(root.Get("content"))
	if len(parts) == 0 {
		return
	}
	if a.role != "" && a.role != role {
		a.Flush()
	}
	a.role = role
	for _, part := range parts {
		if role == "assistant" && gjson.GetBytes(part, "type").String() == "tool_use" {
			a.toolUseParts = append(a.toolUseParts, part)
			continue
		}
		a.content = append(a.content, part)
	}
}

// Flush closes the current role turn while keeping accumulated messages.
func (a *ClaudeMessageAccumulator) Flush() {
	if a.role == "" {
		return
	}
	parts := a.content
	if len(a.toolUseParts) > 0 {
		combined := make([][]byte, 0, len(a.content)+len(a.toolUseParts))
		combined = append(combined, a.content...)
		combined = append(combined, a.toolUseParts...)
		parts = combined
	}
	if len(parts) > 0 {
		message := []byte(`{"role":"","content":[]}`)
		message, _ = sjson.SetBytes(message, "role", a.role)
		message, _ = sjson.SetRawBytes(message, "content", JoinRawArray(parts))
		a.messages = append(a.messages, message)
	}
	a.role = ""
	a.content = nil
	a.toolUseParts = nil
}

// Messages flushes the final turn and returns all accumulated messages.
func (a *ClaudeMessageAccumulator) Messages() [][]byte {
	a.Flush()
	return a.messages
}

const interruptedClaudeToolResultContent = "[operation interrupted by user]"

// RepairDanglingClaudeToolUses synthesizes missing tool_result blocks when an
// assistant tool_use is not closed by the following user/tool message.
// Interrupted tool calls would otherwise 400 on Anthropic Messages (-4003).
//
// Next user/tool messages that omit IDs get synth results prepended while other
// content is kept. A trailing assistant, or a next message that is not
// user/tool, gets a synthetic user message. Complete pairings are unchanged.
// Partial parallel tool_use sets fill only the missing IDs.
func RepairDanglingClaudeToolUses(payload []byte) []byte {
	messages := gjson.GetBytes(payload, "messages")
	if !messages.IsArray() {
		return payload
	}

	messageResults := messages.Array()
	out := make([][]byte, 0, len(messageResults)+2)
	changed := false

	for i := 0; i < len(messageResults); i++ {
		message := messageResults[i]
		out = append(out, []byte(message.Raw))

		ids := claudeAssistantToolUseIDs(message)
		if len(ids) == 0 {
			continue
		}

		if i+1 >= len(messageResults) {
			out = append(out, interruptedClaudeToolResultUser(ids))
			changed = true
			continue
		}

		next := messageResults[i+1]
		nextRole := next.Get("role").String()
		if nextRole != "user" && nextRole != "tool" {
			out = append(out, interruptedClaudeToolResultUser(ids))
			changed = true
			continue
		}

		missing := missingClaudeToolUseIDs(ids, next)
		if len(missing) == 0 {
			continue
		}

		changed = true
		i++
		out = append(out, prependClaudeToolResults(next, missing))
	}

	if !changed {
		return payload
	}
	updated, err := sjson.SetRawBytes(payload, "messages", JoinRawArray(out))
	if err != nil {
		return payload
	}
	return updated
}

func claudeAssistantToolUseIDs(message gjson.Result) []string {
	if message.Get("role").String() != "assistant" {
		return nil
	}
	content := message.Get("content")
	if !content.IsArray() {
		return nil
	}
	ids := make([]string, 0)
	content.ForEach(func(_, part gjson.Result) bool {
		if part.Get("type").String() != "tool_use" {
			return true
		}
		if id := part.Get("id").String(); id != "" {
			ids = append(ids, id)
		}
		return true
	})
	return ids
}

func missingClaudeToolUseIDs(ids []string, next gjson.Result) []string {
	provided := make(map[string]struct{}, len(ids))
	content := next.Get("content")
	if content.IsArray() {
		content.ForEach(func(_, part gjson.Result) bool {
			if part.Get("type").String() != "tool_result" {
				return true
			}
			id := part.Get("tool_use_id").String()
			if id == "" {
				id = part.Get("id").String()
			}
			if id != "" {
				provided[id] = struct{}{}
			}
			return true
		})
	}
	missing := make([]string, 0)
	for _, id := range ids {
		if _, ok := provided[id]; !ok {
			missing = append(missing, id)
		}
	}
	return missing
}

func prependClaudeToolResults(message gjson.Result, missingIDs []string) []byte {
	parts := interruptedClaudeToolResultParts(missingIDs)
	content := message.Get("content")
	switch {
	case content.Type == gjson.String:
		if text := content.String(); strings.TrimSpace(text) != "" {
			part := []byte(`{"type":"text","text":""}`)
			part, _ = sjson.SetBytes(part, "text", text)
			parts = append(parts, part)
		}
	case content.IsArray():
		content.ForEach(func(_, part gjson.Result) bool {
			if part.IsObject() {
				parts = append(parts, []byte(part.Raw))
			}
			return true
		})
	}
	updated := []byte(message.Raw)
	updated, _ = sjson.SetRawBytes(updated, "content", JoinRawArray(parts))
	return updated
}

func interruptedClaudeToolResultUser(ids []string) []byte {
	message := []byte(`{"role":"user","content":[]}`)
	message, _ = sjson.SetRawBytes(message, "content", JoinRawArray(interruptedClaudeToolResultParts(ids)))
	return message
}

func interruptedClaudeToolResultParts(ids []string) [][]byte {
	parts := make([][]byte, 0, len(ids))
	for _, id := range ids {
		part := []byte(`{"type":"tool_result","tool_use_id":"","content":"","is_error":true}`)
		part, _ = sjson.SetBytes(part, "tool_use_id", id)
		part, _ = sjson.SetBytes(part, "content", interruptedClaudeToolResultContent)
		parts = append(parts, part)
	}
	return parts
}

// AlignClaudeToolResults orders tool_result blocks by the preceding tool_use IDs.
// Other content blocks retain their relative order after the tool results. If a
// complete one-to-one match is unavailable, the original content is returned.
func AlignClaudeToolResults(content gjson.Result, toolUseIDs []string) gjson.Result {
	if !content.IsArray() || len(toolUseIDs) == 0 {
		return content
	}

	parts := content.Array()
	toolResults := make([]gjson.Result, 0, len(toolUseIDs))
	otherParts := make([]gjson.Result, 0, len(parts))
	for _, part := range parts {
		if part.Get("type").String() == "tool_result" {
			toolResults = append(toolResults, part)
			continue
		}
		otherParts = append(otherParts, part)
	}
	if len(toolResults) != len(toolUseIDs) {
		return content
	}

	ordered := make([][]byte, 0, len(parts))
	used := make([]bool, len(toolResults))
	for _, toolUseID := range toolUseIDs {
		matched := -1
		for resultIndex, toolResult := range toolResults {
			if !used[resultIndex] && toolUseID != "" && toolResult.Get("tool_use_id").String() == toolUseID {
				matched = resultIndex
				break
			}
		}
		if matched < 0 {
			return content
		}
		used[matched] = true
		ordered = append(ordered, []byte(toolResults[matched].Raw))
	}
	for _, part := range otherParts {
		ordered = append(ordered, []byte(part.Raw))
	}
	return gjson.ParseBytes(JoinRawArray(ordered))
}

func claudeMessageContentParts(content gjson.Result) [][]byte {
	if !content.Exists() || content.Type == gjson.Null {
		return nil
	}
	if content.Type == gjson.String {
		if content.String() == "" {
			return nil
		}
		part := []byte(`{"type":"text","text":""}`)
		part, _ = sjson.SetBytes(part, "text", content.String())
		return [][]byte{part}
	}
	if !content.IsArray() {
		return nil
	}
	parts := make([][]byte, 0, len(content.Array()))
	content.ForEach(func(_, part gjson.Result) bool {
		if part.IsObject() {
			parts = append(parts, []byte(part.Raw))
		}
		return true
	})
	return parts
}
