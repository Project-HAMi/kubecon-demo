---
name: Bug Report
about: Report a problem encountered while using KubeCon demo
labels: bug
---

<!-- Please use this template while reporting a bug and provide as much info as possible. Not doing so may result in your bug not being addressed in a timely manner. Thanks!
-->

**What happened**:

**What you expected to happen**:

**How to reproduce it (as minimally and precisely as possible)**:

**Anything else we need to know?**:

- The failing `make` target and minimal command output
- Relevant Helmfile values or chart manifest sections
- Relevant, time-bounded HAMi, scheduler, workload, and kubelet log excerpts
- Sanitized pod status, events, and GPU allocation output
- Relevant `test_vllm.py` traceback or API response excerpt

Before posting, include only relevant, time-bounded excerpts and remove or mask credentials, tokens, private keys, certificates, device identifiers, node or host names, workload identifiers, and internal image names.

**Environment**:
- kubecon-demo version or commit:
- HAMi and HAMi-core versions:
- Kubernetes distribution and version:
- Helm and Helmfile versions:
- GPU model and driver or runtime version:
- vLLM and workload image tags:
- Python version:
- Cloud or local cluster type:
- Others:
