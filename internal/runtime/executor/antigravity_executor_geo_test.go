package executor

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
	"github.com/tidwall/gjson"
)

type mockGeoRoundTripperProvider struct {
	trippers map[string]http.RoundTripper
}

func (m *mockGeoRoundTripperProvider) RoundTripperFor(auth *cliproxyauth.Auth) http.RoundTripper {
	if auth == nil {
		return nil
	}
	return m.trippers[auth.ID]
}

type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestAntigravityGeoError_Classification(t *testing.T) {
	t.Run("location_rejected_detail", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "User location is not supported for the API use.",
				"status": "FAILED_PRECONDITION",
				"details": [
					{
						"@type": "type.googleapis.com/google.rpc.ErrorInfo",
						"reason": "LOCATION_REJECTED",
						"domain": "googleapis.com"
					}
				]
			}
		}`)
		if !isAntigravityGeoError(http.StatusBadRequest, body) {
			t.Fatal("expected isAntigravityGeoError to be true for LOCATION_REJECTED")
		}
	})

	t.Run("empty_reason_detail", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "User location is not supported for the API use.",
				"status": "FAILED_PRECONDITION",
				"details": [
					{
						"@type": "type.googleapis.com/google.rpc.ErrorInfo",
						"reason": "",
						"domain": "googleapis.com"
					}
				]
			}
		}`)
		if !isAntigravityGeoError(http.StatusBadRequest, body) {
			t.Fatal("expected isAntigravityGeoError to be true for empty reason")
		}
	})

	t.Run("no_details", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "User location is not supported for the API use.",
				"status": "FAILED_PRECONDITION"
			}
		}`)
		if !isAntigravityGeoError(http.StatusBadRequest, body) {
			t.Fatal("expected isAntigravityGeoError to be true for missing details")
		}
	})

	t.Run("unauthenticated_detail_must_not_match", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "User location is not supported for the API use.",
				"status": "FAILED_PRECONDITION",
				"details": [
					{
						"@type": "type.googleapis.com/google.rpc.ErrorInfo",
						"reason": "UNAUTHENTICATED",
						"domain": "googleapis.com"
					}
				]
			}
		}`)
		if isAntigravityGeoError(http.StatusBadRequest, body) {
			t.Fatal("expected isAntigravityGeoError to be false for UNAUTHENTICATED")
		}
	})

	t.Run("conflicting_status_must_not_match", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "User location is not supported for the API use.",
				"status": "RESOURCE_EXHAUSTED",
				"details": [{"reason": "LOCATION_REJECTED"}]
			}
		}`)
		if isAntigravityGeoError(http.StatusBadRequest, body) {
			t.Fatal("expected isAntigravityGeoError to be false for conflicting status")
		}
	})

	t.Run("conflicting_http_status_code", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "User location is not supported for the API use.",
				"status": "FAILED_PRECONDITION",
				"details": [{"reason": "LOCATION_REJECTED"}]
			}
		}`)
		if isAntigravityGeoError(http.StatusInternalServerError, body) {
			t.Fatal("expected isAntigravityGeoError to be false for 500 status code")
		}
	})

	t.Run("unrelated_error_message", func(t *testing.T) {
		body := []byte(`{
			"error": {
				"code": 400,
				"message": "Invalid argument provided.",
				"status": "FAILED_PRECONDITION"
			}
		}`)
		if isAntigravityGeoError(http.StatusBadRequest, body) {
			t.Fatal("expected isAntigravityGeoError to be false for unrelated message")
		}
	})
}

func TestAntigravityGeoError_CooldownAndSwitch(t *testing.T) {
	resetAntigravityCreditsRetryState()
	cfg := &config.Config{DisableCooling: false}
	manager := cliproxyauth.NewManager(nil, nil, nil)
	manager.SetRetryConfig(0, 0, 2)

	executor := NewAntigravityExecutor(cfg)
	manager.RegisterExecutor(executor)

	baseID := uuid.NewString()
	auth1 := &cliproxyauth.Auth{ID: baseID + "-geo-1", Provider: "antigravity", Metadata: map[string]any{"access_token": "token-1", "project_id": "p-1", "expires_at": time.Now().Add(time.Hour).Unix()}}
	auth2 := &cliproxyauth.Auth{ID: baseID + "-geo-2", Provider: "antigravity", Metadata: map[string]any{"access_token": "token-2", "project_id": "p-2", "expires_at": time.Now().Add(time.Hour).Unix()}}

	model := "gemini-2.5-flash"
	reg := registry.GetGlobalRegistry()
	reg.RegisterClient(auth1.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	reg.RegisterClient(auth2.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() {
		reg.UnregisterClient(auth1.ID)
		reg.UnregisterClient(auth2.ID)
		resetAntigravityCreditsRetryState()
	})

	geoBody := `{"error":{"code":400,"message":"User location is not supported for the API use.","status":"FAILED_PRECONDITION","details":[{"reason":"LOCATION_REJECTED"}]}}`
	successBody := `{"candidates":[{"content":{"parts":[{"text":"hello from auth2"}]},"finishReason":"STOP"}]}`

	auth1Called := false
	auth2Called := false

	provider := &mockGeoRoundTripperProvider{
		trippers: map[string]http.RoundTripper{
			auth1.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				auth1Called = true
				return &http.Response{
					StatusCode: http.StatusBadRequest,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader([]byte(geoBody))),
				}, nil
			}),
			auth2.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				auth2Called = true
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader([]byte(successBody))),
				}, nil
			}),
		},
	}
	manager.SetRoundTripperProvider(provider)

	if _, err := manager.Register(context.Background(), auth1); err != nil {
		t.Fatalf("register auth1: %v", err)
	}
	if _, err := manager.Register(context.Background(), auth2); err != nil {
		t.Fatalf("register auth2: %v", err)
	}

	payload := []byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`)
	resp, err := manager.Execute(context.Background(), []string{"antigravity"}, cliproxyexecutor.Request{
		Model:   model,
		Payload: payload,
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatGemini})
	if err != nil {
		t.Fatalf("expected failover to auth2 on geo error, got error: %v", err)
	}

	if !auth1Called {
		t.Fatal("expected auth1 to be called")
	}
	if !auth2Called {
		t.Fatal("expected auth2 to be called after geo failover")
	}
	if len(resp.Payload) == 0 {
		t.Fatal("expected non-empty response payload from auth2")
	}

	// Verify auth1 is under cooldown
	if inCool, _, _ := antigravityIsInShortCooldownRequired(context.Background(), auth1, model, time.Now()); !inCool {
		t.Fatal("expected auth1 to be marked with short cooldown")
	}
}

func TestAntigravityGeoError_UnauthenticatedDoesNotCoolOrSwitch(t *testing.T) {
	resetAntigravityCreditsRetryState()
	cfg := &config.Config{DisableCooling: false}
	manager := cliproxyauth.NewManager(nil, nil, nil)
	manager.SetRetryConfig(0, 0, 2)

	executor := NewAntigravityExecutor(cfg)
	manager.RegisterExecutor(executor)

	baseID := uuid.NewString()
	auth1 := &cliproxyauth.Auth{ID: baseID + "-unauth-1", Provider: "antigravity", Metadata: map[string]any{"access_token": "token-1", "project_id": "p-1", "expires_at": time.Now().Add(time.Hour).Unix()}}
	auth2 := &cliproxyauth.Auth{ID: baseID + "-unauth-2", Provider: "antigravity", Metadata: map[string]any{"access_token": "token-2", "project_id": "p-2", "expires_at": time.Now().Add(time.Hour).Unix()}}

	model := "gemini-2.5-flash"
	reg := registry.GetGlobalRegistry()
	reg.RegisterClient(auth1.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	reg.RegisterClient(auth2.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() {
		reg.UnregisterClient(auth1.ID)
		reg.UnregisterClient(auth2.ID)
		resetAntigravityCreditsRetryState()
	})

	unauthBody := `{"error":{"code":400,"message":"User location is not supported for the API use.","status":"FAILED_PRECONDITION","details":[{"reason":"UNAUTHENTICATED"}]}}`

	auth1Called := false
	auth2Called := false

	provider := &mockGeoRoundTripperProvider{
		trippers: map[string]http.RoundTripper{
			auth1.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				auth1Called = true
				return &http.Response{
					StatusCode: http.StatusBadRequest,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader([]byte(unauthBody))),
				}, nil
			}),
			auth2.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				auth2Called = true
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader([]byte(`{}`))),
				}, nil
			}),
		},
	}
	manager.SetRoundTripperProvider(provider)

	if _, err := manager.Register(context.Background(), auth1); err != nil {
		t.Fatalf("register auth1: %v", err)
	}
	if _, err := manager.Register(context.Background(), auth2); err != nil {
		t.Fatalf("register auth2: %v", err)
	}

	payload := []byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`)
	_, err := manager.Execute(context.Background(), []string{"antigravity"}, cliproxyexecutor.Request{
		Model:   model,
		Payload: payload,
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatGemini})
	if err == nil {
		t.Fatal("expected request to fail with unauthenticated error, got nil")
	}

	if !auth1Called {
		t.Fatal("expected auth1 to be called")
	}
	if auth2Called {
		t.Fatal("expected auth2 NOT to be called (UNAUTHENTICATED should not switch)")
	}

	// Verify auth1 is NOT cooled down
	if inCool, _, _ := antigravityIsInShortCooldownRequired(context.Background(), auth1, model, time.Now()); inCool {
		t.Fatal("auth1 should NOT be marked with short cooldown")
	}
}

func TestAntigravityGeoError_ExhaustedReturns403WithoutRetryAfter(t *testing.T) {
	resetAntigravityCreditsRetryState()
	cfg := &config.Config{DisableCooling: false}
	manager := cliproxyauth.NewManager(nil, nil, nil)
	manager.SetRetryConfig(0, 0, 2)

	executor := NewAntigravityExecutor(cfg)
	manager.RegisterExecutor(executor)

	baseID := uuid.NewString()
	auth1 := &cliproxyauth.Auth{ID: baseID + "-exhaust-1", Provider: "antigravity", Metadata: map[string]any{"access_token": "token-1", "project_id": "p-1", "expires_at": time.Now().Add(time.Hour).Unix()}}

	model := "gemini-2.5-flash"
	reg := registry.GetGlobalRegistry()
	reg.RegisterClient(auth1.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() {
		reg.UnregisterClient(auth1.ID)
		resetAntigravityCreditsRetryState()
	})

	geoBody := `{"error":{"code":400,"message":"User location is not supported for the API use.","status":"FAILED_PRECONDITION","details":[{"reason":"LOCATION_REJECTED"}]}}`

	provider := &mockGeoRoundTripperProvider{
		trippers: map[string]http.RoundTripper{
			auth1.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusBadRequest,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader([]byte(geoBody))),
				}, nil
			}),
		},
	}
	manager.SetRoundTripperProvider(provider)

	if _, err := manager.Register(context.Background(), auth1); err != nil {
		t.Fatalf("register auth1: %v", err)
	}

	payload := []byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`)
	_, err := manager.Execute(context.Background(), []string{"antigravity"}, cliproxyexecutor.Request{
		Model:   model,
		Payload: payload,
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatGemini})
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	type statusCoder interface {
		StatusCode() int
	}
	type retryAfterGetter interface {
		RetryAfter() *time.Duration
	}

	var sc statusCoder
	if scErr, ok := err.(statusCoder); ok {
		sc = scErr
	} else {
		t.Fatalf("expected error to implement StatusCode(), got %T", err)
	}

	if code := sc.StatusCode(); code != http.StatusForbidden {
		t.Fatalf("expected status code 403, got %d", code)
	}

	if raGetter, ok := err.(retryAfterGetter); ok {
		if ra := raGetter.RetryAfter(); ra != nil {
			t.Fatalf("expected nil Retry-After, got %v", *ra)
		}
	}

	safeHeaders := cliproxyauth.SafeResponseHeaders(err)
	if safeHeaders.Get("Retry-After") != "" {
		t.Fatalf("SafeResponseHeaders has Retry-After %q, want empty", safeHeaders.Get("Retry-After"))
	}

	errStr := err.Error()
	parsed := gjson.Parse(errStr)
	if parsed.Get("error.type").String() != "permission_error" {
		t.Fatalf("error.type = %q, want permission_error; err = %s", parsed.Get("error.type").String(), errStr)
	}
	if parsed.Get("error.code").String() != "regional_restriction" {
		t.Fatalf("error.code = %q, want regional_restriction; err = %s", parsed.Get("error.code").String(), errStr)
	}
}

func TestAntigravityGeoError_StreamCooldownAndSwitch(t *testing.T) {
	resetAntigravityCreditsRetryState()
	cfg := &config.Config{DisableCooling: false}
	manager := cliproxyauth.NewManager(nil, nil, nil)
	manager.SetRetryConfig(0, 0, 2)

	executor := NewAntigravityExecutor(cfg)
	manager.RegisterExecutor(executor)

	baseID := uuid.NewString()
	auth1 := &cliproxyauth.Auth{
		ID:       baseID + "-stream-geo-1",
		Provider: "antigravity",
		Metadata: map[string]any{"access_token": "token-1", "project_id": "p-1", "expires_at": time.Now().Add(time.Hour).Unix()},
	}
	auth2 := &cliproxyauth.Auth{
		ID:       baseID + "-stream-geo-2",
		Provider: "antigravity",
		Metadata: map[string]any{"access_token": "token-2", "project_id": "p-2", "expires_at": time.Now().Add(time.Hour).Unix()},
	}

	model := "gemini-2.5-flash"
	reg := registry.GetGlobalRegistry()
	reg.RegisterClient(auth1.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	reg.RegisterClient(auth2.ID, "antigravity", []*registry.ModelInfo{{ID: model}})
	t.Cleanup(func() {
		reg.UnregisterClient(auth1.ID)
		reg.UnregisterClient(auth2.ID)
		resetAntigravityCreditsRetryState()
	})

	geoBody := `{"error":{"code":400,"message":"User location is not supported for the API use.","status":"FAILED_PRECONDITION","details":[{"reason":"LOCATION_REJECTED"}]}}`
	successChunk := "data: " + `{"response":{"candidates":[{"content":{"parts":[{"text":"hello stream"}]},"finishReason":"STOP"}]}}` + "\n\n"

	auth1Called := false
	auth2Called := false

	provider := &mockGeoRoundTripperProvider{
		trippers: map[string]http.RoundTripper{
			auth1.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				auth1Called = true
				return &http.Response{
					StatusCode: http.StatusBadRequest,
					Header:     make(http.Header),
					Body:       io.NopCloser(bytes.NewReader([]byte(geoBody))),
				}, nil
			}),
			auth2.ID: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				auth2Called = true
				respHeader := make(http.Header)
				respHeader.Set("Content-Type", "text/event-stream")
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     respHeader,
					Body:       io.NopCloser(bytes.NewReader([]byte(successChunk))),
				}, nil
			}),
		},
	}
	manager.SetRoundTripperProvider(provider)

	if _, err := manager.Register(context.Background(), auth1); err != nil {
		t.Fatalf("register auth1: %v", err)
	}
	if _, err := manager.Register(context.Background(), auth2); err != nil {
		t.Fatalf("register auth2: %v", err)
	}

	payload := []byte(`{"contents":[{"parts":[{"text":"hi"}]}]}`)
	streamResp, err := manager.ExecuteStream(context.Background(), []string{"antigravity"}, cliproxyexecutor.Request{
		Model:   model,
		Payload: payload,
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatGemini})
	if err != nil {
		t.Fatalf("expected stream failover to auth2, got error: %v", err)
	}

	if !auth1Called {
		t.Fatal("expected auth1 to be called")
	}
	if !auth2Called {
		t.Fatal("expected auth2 to be called after geo failover")
	}

	// Read chunks
	hasChunk := false
	for chunk := range streamResp.Chunks {
		if len(chunk.Payload) > 0 {
			hasChunk = true
		}
	}
	if !hasChunk {
		t.Fatal("expected chunks from auth2 stream")
	}

	// Verify auth1 is under cooldown
	inCool, _, _ := antigravityIsInShortCooldownRequired(context.Background(), auth1, model, time.Now())
	if !inCool {
		t.Fatal("expected auth1 to be marked with short cooldown")
	}
}
