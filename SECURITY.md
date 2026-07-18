# Security policy

## Supported version

Security fixes target the latest tagged release and `main`.

## Reporting

Do not open a public issue for a vulnerability or include credentials, private media, or personal
data in a report. Use GitHub's private vulnerability reporting for this repository.

## Deployment boundary

The development server binds to loopback by default and has no built-in user authentication.
Before any network exposure, place it behind an authenticated TLS reverse proxy, apply request
rate limits, isolate the media volume, and restrict outbound model-server access. Never deploy the
demo as an autonomous high-stakes decision system.

