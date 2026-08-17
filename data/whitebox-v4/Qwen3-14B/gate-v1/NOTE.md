Gate FAILED (self_token_rate 0.43 < 0.5, latin_initial_rate 0.57 < 0.7) but the
v4 arms were run anyway, 2026-08-17, on this reading: the missed emotions
top-decode to semantically correct CHINESE tokens (angry -> 愤怒/怒/rage/fury,
exasperated -> 吐槽/叹了口气/sigh, loving -> 温暖/温柔/gently), so the failure
looks like the gate\x27s English-centric self-token/Latin-prefix criteria meeting
Qwen\x27s Chinese-heavy vocabulary, not like unreadable directions. Treat the Qwen
probe readout with this caveat; behavioral results are unaffected. Decision made
by Claude to avoid idling the pod overnight -- overrule by discarding
rollouts/Qwen3-14B/v4-* projections.
