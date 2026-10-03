package chat_completions

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertGeminiResponseToOpenAICompletionTokensIncludeThoughts(t *testing.T) {
	var param any
	chunk := []byte(`{"usageMetadata":{"promptTokenCount":16,"candidatesTokenCount":5,"thoughtsTokenCount":42,"totalTokenCount":63}}`)

	result := ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk, &param)
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	completionTokens := gjson.GetBytes(result[0], "usage.completion_tokens")
	if !completionTokens.Exists() || completionTokens.Int() != 47 {
		t.Fatalf("completion_tokens = %s, want present with value 47. Output: %s", completionTokens.Raw, result[0])
	}
}

func TestConvertGeminiResponseToOpenAINonStreamCompletionTokensIncludeThoughts(t *testing.T) {
	response := []byte(`{"usageMetadata":{"promptTokenCount":16,"thoughtsTokenCount":42,"totalTokenCount":58}}`)

	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)
	completionTokens := gjson.GetBytes(result, "usage.completion_tokens")
	if !completionTokens.Exists() || completionTokens.Int() != 42 {
		t.Fatalf("completion_tokens = %s, want present with value 42. Output: %s", completionTokens.Raw, result)
	}
}

func TestGeminiFinishReasonOnlyOnFinalChunk(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"list_dir","args":{"path":"C:/"}}}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"}}`)
	result1 := ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk1, &param)
	if len(result1) != 1 {
		t.Fatalf("expected 1 result from chunk1, got %d", len(result1))
	}
	fr1 := gjson.GetBytes(result1[0], "choices.0.finish_reason")
	if fr1.Exists() && fr1.String() != "" && fr1.Type.String() != "Null" {
		t.Fatalf("expected null finish_reason on tool chunk, got %v", fr1.String())
	}

	chunk2 := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"list_dir","args":{"path":"D:/"}}}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"}}`)
	ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk2, &param)

	chunk3 := []byte(`{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`)
	result3 := ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk3, &param)
	if len(result3) != 1 {
		t.Fatalf("expected 1 result from chunk3, got %d", len(result3))
	}
	fr3 := gjson.GetBytes(result3[0], "choices.0.finish_reason").String()
	if fr3 != "tool_calls" {
		t.Fatalf("expected finish_reason tool_calls, got %s", fr3)
	}
	nfr3 := gjson.GetBytes(result3[0], "choices.0.native_finish_reason").String()
	if nfr3 != "stop" {
		t.Fatalf("expected native_finish_reason stop, got %s", nfr3)
	}
}

func TestConvertGeminiResponseToOpenAINonStream_EmptyTextProducesEmptyString(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"text":""},{"text":"","thought":true}]},"finishReason":"STOP"}]}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "" || content.Type == gjson.Null {
		t.Fatalf("expected content to be empty string \"\", got %v (type %v)", content.Value(), content.Type)
	}

	reasoning := gjson.GetBytes(result, "choices.0.message.reasoning_content")
	if !reasoning.Exists() || reasoning.String() != "" || reasoning.Type == gjson.Null {
		t.Fatalf("expected reasoning_content to be empty string \"\", got %v (type %v)", reasoning.Value(), reasoning.Type)
	}
}

// Gemini 3.5 Transcribe returns its transcript in an audioTranscription part instead
// of the regular text part. Without handling it, the transcript is silently dropped
// and the OpenAI client sees an empty assistant message.
func TestConvertGeminiResponseToOpenAINonStream_AudioTranscriptionPartProducesContent(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"text":""},{"audioTranscription":{"text":"Hello world"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":185,"totalTokenCount":185}}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "Hello world" {
		t.Fatalf("expected content to carry the transcript, got %v. Output: %s", content.Raw, result)
	}
}

func TestConvertGeminiResponseToOpenAINonStream_SingleAudioTranscriptionPart(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"audioTranscription":{"text":"Single transcription part"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":185,"totalTokenCount":185}}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "Single transcription part" {
		t.Fatalf("expected content to carry the transcript, got %v. Output: %s", content.Raw, result)
	}
}

func TestConvertGeminiResponseToOpenAI_AudioTranscriptionPartStreamsContent(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk := []byte(`{"candidates":[{"content":{"parts":[{"text":""},{"audioTranscription":{"text":"Testing one two three."}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":185,"candidatesTokenCount":8,"totalTokenCount":193}}`)
	result := ConvertGeminiResponseToOpenAI(ctx, "model", nil, nil, chunk, &param)
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}

	content := gjson.GetBytes(result[0], "choices.0.delta.content")
	if !content.Exists() || content.String() != "Testing one two three." {
		t.Fatalf("expected delta.content to carry the transcript, got %v. Output: %s", content.Raw, result[0])
	}
}

func TestConvertGeminiResponseToOpenAINonStream_TextPrecedenceOverAudioTranscription(t *testing.T) {
	response := []byte(`{"candidates":[{"content":{"parts":[{"text":"explicit text","audioTranscription":{"text":"ignored transcription"}}]},"finishReason":"STOP"}]}`)
	result := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, response, nil)

	content := gjson.GetBytes(result, "choices.0.message.content")
	if !content.Exists() || content.String() != "explicit text" {
		t.Fatalf("expected content to prioritize explicit text, got %v. Output: %s", content.Raw, result)
	}
}

func TestGeminiOpenAIMaxTokensHandling(t *testing.T) {
	t.Run("nonstream_thinking_only", func(t *testing.T) {
		resp := []byte(`{"candidates":[{"content":{"parts":[{"text":"thinking...","thought":true}]},"finishReason":"MAX_TOKENS"}]}`)
		out := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, resp, nil)
		if got := gjson.GetBytes(out, "choices.0.finish_reason").String(); got != "stop" {
			t.Fatalf("finish_reason = %q, want stop: %s", got, out)
		}
	})

	t.Run("nonstream_visible_text", func(t *testing.T) {
		resp := []byte(`{"candidates":[{"content":{"parts":[{"text":"visible text"}]},"finishReason":"MAX_TOKENS"}]}`)
		out := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, resp, nil)
		if got := gjson.GetBytes(out, "choices.0.finish_reason").String(); got != "max_tokens" {
			t.Fatalf("finish_reason = %q, want max_tokens: %s", got, out)
		}
	})

	t.Run("nonstream_tool_calls_priority", func(t *testing.T) {
		resp := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"test"}}}]},"finishReason":"MAX_TOKENS"}]}`)
		out := ConvertGeminiResponseToOpenAINonStream(context.Background(), "model", nil, nil, resp, nil)
		if got := gjson.GetBytes(out, "choices.0.finish_reason").String(); got != "tool_calls" {
			t.Fatalf("finish_reason = %q, want tool_calls: %s", got, out)
		}
	})

	t.Run("stream_thinking_only", func(t *testing.T) {
		var param any
		chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"text":"thinking...","thought":true}]}}]}`)
		chunk2 := []byte(`{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`)
		_ = ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk1, &param)
		out := ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk2, &param)
		out = append(out, ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, []byte("[DONE]"), &param)...)
		joined := string(bytes.Join(out, nil))
		if !strings.Contains(joined, `"finish_reason":"stop"`) {
			t.Fatalf("stream should emit finish_reason stop: %s", joined)
		}
	})

	t.Run("stream_visible_text", func(t *testing.T) {
		var param any
		chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"text":"visible text"}]}}]}`)
		chunk2 := []byte(`{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`)
		_ = ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk1, &param)
		out := ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk2, &param)
		out = append(out, ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, []byte("[DONE]"), &param)...)
		joined := string(bytes.Join(out, nil))
		if !strings.Contains(joined, `"finish_reason":"max_tokens"`) {
			t.Fatalf("stream should emit finish_reason max_tokens: %s", joined)
		}
	})

	t.Run("stream_tool_calls_priority", func(t *testing.T) {
		var param any
		chunk1 := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"test"}}}]}}]}`)
		chunk2 := []byte(`{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}}`)
		_ = ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk1, &param)
		out := ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, chunk2, &param)
		out = append(out, ConvertGeminiResponseToOpenAI(context.Background(), "model", nil, nil, []byte("[DONE]"), &param)...)
		joined := string(bytes.Join(out, nil))
		if !strings.Contains(joined, `"finish_reason":"tool_calls"`) {
			t.Fatalf("stream should emit finish_reason tool_calls: %s", joined)
		}
	})
}
