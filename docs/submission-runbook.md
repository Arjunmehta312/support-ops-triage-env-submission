# Submission Runbook

This runbook is the final operator checklist for the `support_ops_triage_env` hackathon submission.

## Canonical Deployment Targets

- Main workspace repo: `Arjunmehta312/meta-pytorch-hackathon`
- Submission repo: `Arjunmehta312/support-ops-triage-env-submission`
- Hugging Face Space: `Arjunmehta312/support-ops-triage-env`

## Pre-Submit Checks

Run from `support_ops_triage_env/`.

1. Validate OpenEnv contract:

```bash
openenv validate .
```

2. Validate local submission package behavior:

```bash
python scripts/pre_submission_validate.py
```

3. Validate deployed Space behavior:

```bash
OPENENV_BASE_URL=https://arjunmehta312-support-ops-triage-env.hf.space python scripts/pre_submission_validate.py
```

4. Check live runtime health snapshot:

```bash
python scripts/check_space_status.py --owner Arjunmehta312 --space support-ops-triage-env
```

## UI Smoke Test

Open:

- https://arjunmehta312-support-ops-triage-env.hf.space/web

Then:

1. Click `Reset`
2. Set `operation=classify`, `ticket_id=E-101`, `classification=billing`
3. Click `Step`
4. Verify no UI error and `Raw JSON response` updates

## Acceptance Criteria

Submission is ready when all are true:

- `runtime_api`, `/health`, and `/tasks` checks are OK
- Remote pre-submission validator returns `status: ok`
- `/web` route is reachable
- `reset` and `step` return HTTP 200

## Notes

- HF may show transitional stage labels (`BUILDING`, `APP_STARTING`) during rollout.
- Use endpoint/validator checks as source of truth.
- Do not commit secrets (`HF_TOKEN`, API keys) into repository files.
