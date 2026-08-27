# Workshop Candidate Manifest

The holdout campaign accepts preserved candidate files only. It does not call a
model API. Candidate generation must produce a JSON manifest with this shape:

```json
{
  "schema_version": 2,
  "task_selection_manifest_sha256": "<sha256>",
  "provider": "<provider>",
  "configured_model_string": "<exact configured string>",
  "provider_response_model_fields_preserved": true,
  "prompt_version": "workshop2026_v1",
  "tasks": {
    "<task_id>": [
      {
        "candidate_id": "candidate_000",
        "path": "/absolute/durable/path/candidate_000.py",
        "sha256": "<sha256>",
        "prompt_path": "/absolute/durable/path/candidate_000.prompt.txt",
        "prompt_sha256": "<sha256>",
        "raw_response_path": "/absolute/durable/path/candidate_000.response.txt",
        "raw_response_sha256": "<sha256>",
        "metadata_path": "/absolute/durable/path/candidate_000.metadata.json",
        "metadata_sha256": "<sha256>",
        "provider_response_model": "<provider-returned field or not_preserved>"
      }
    ]
  }
}
```

Exactly three candidates are required per frozen task. The campaign verifies
the candidate, prompt, raw response, and backend-metadata path and checksum
before screening. Provider-returned model metadata is mandatory for this new
campaign; `not_preserved` is not an acceptable value. Screening freezes one
candidate per task; confirmation refuses to continue if that winner mapping
changes.
