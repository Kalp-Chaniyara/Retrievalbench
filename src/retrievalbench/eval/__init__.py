import os

# DeepEval phones home to PostHog on import/measure. Behind a TLS-intercepting
# proxy those uploads fail certificate verification and hang, stalling evaluation
# until DeepEval's ~88s per-attempt timeout trips (the RetryError we hit). This
# package __init__ runs before `eval.metric` imports deepeval, so opting out here
# guarantees the flag is set first. setdefault: an explicit env override wins.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")
