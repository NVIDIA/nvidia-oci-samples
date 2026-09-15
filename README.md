# NVIDIA GPU Accelerated Application Samples in Oracle Cloud Infrastructure (OCI)

**Table Of Contents**
- [Description](#description)
- [Support Level](#support-level)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Samples](#samples)
- [Usage](#usage)
- [Additional Resources](#additional-resources)
- [Known Issues](#known-issues)
- [Contributions](#contributions)
- [Support](#support)
- [License](#license)
- [Maintainers](#maintainers)

## Description

This repository maintains sample applications designed for NVIDIA software tools integrated with Oracle Cloud Infrastructure (OCI).

For select demonstrations, the sample code is contained within this repository. For others, we reference and link to exceptional demonstrations available outside of this repository.

## Support Level

These samples are provided as community examples and are not covered by NVIDIA Enterprise Support. They are intended as reference implementations to help users get started with NVIDIA software on OCI. See [SUPPORT](#support) below for how to get help.

## Requirements

- An active [Oracle Cloud Infrastructure (OCI)](https://www.oracle.com/cloud/) account with permissions to create the resources used by a given sample
- Access to NVIDIA GPU shapes in your target OCI region (e.g., A10, A100, H100, H200)
- The OCI CLI installed and configured, or access to the OCI Console
- Additional, sample-specific prerequisites are documented in each sample's own README

> Samples under [`inference/dgx-spark/`](./inference/dgx-spark) run on local DGX Spark hardware and require no OCI account or cloud GPU shapes.

## Quickstart

1. Clone this repository:
   ```bash
   git clone https://github.com/NVIDIA/nvidia-oci-samples.git
   cd nvidia-oci-samples
   ```
2. Browse the [Samples](#samples) section below and pick the one that matches your use case.
3. Follow the README inside that sample's directory for setup and run instructions.

## Samples

Samples are organized first by use case, then by OCI service or deployment platform, then by NVIDIA library or stack.

### Agentic

- OKE / AIQ: [Deploy NVIDIA AIQ 2.0](agentic/oke/aiq/aiq-2.0)

### Inference

- OKE / vLLM: [Nemotron Lightning endpoint](inference/oke/vllm/nemotron-lightning-endpoint)
- Compute / Multi-runtime: [FLUX.1 inference benchmarking](inference/compute/flux-1-benchmarking)
- Compute / Model Optimizer: [FLUX.1 NVFP4 quantization](inference/compute/model-optimizer/flux-1-quantization)
- DGX Spark / NeMo Switchyard: [Route requests between Nemotron and Qwen](inference/dgx-spark/nemo-switchyard)
- DGX Spark / vLLM: [Nemotron Lightning endpoint](inference/dgx-spark/vllm/nemotron-lightning-endpoint)

### Industry Solutions

- Compute / Nemotron: [Agentic multimodal expense intelligence](industry-solutions/compute/nemotron/agentic-multimodal-expense-intelligence)
- OKE / AIQ: [Deploy NVIDIA AIQ 2.0](industry-solutions/oke/aiq/aiq-2.0)

Additional samples will be added over time.

## Usage

Each sample directory contains its own README with detailed deployment and usage instructions. In general:

1. Provision the required OCI infrastructure (cluster, compute, networking, storage) as described in the sample.
2. Deploy the NVIDIA software components.
3. Run the included workloads or applications.
4. Clean up the resources when you are done to avoid unnecessary charges.

## Additional Resources

- [NVIDIA on Oracle Cloud Infrastructure](https://www.nvidia.com/en-us/data-center/oracle-cloud/)
- [Oracle Cloud Infrastructure Documentation](https://docs.oracle.com/en-us/iaas/Content/home.htm)
- [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/)

## Known Issues

None at this time.

## Contributions

Contributions are welcome. Developers can contribute by opening a [pull request](https://help.github.com/en/articles/about-pull-requests) and agreeing to the terms in [CONTRIBUTING.MD](CONTRIBUTING.MD) and [CLA.MD](CLA.MD).

## Support

For questions or issues:
- Open a [GitHub issue](https://github.com/NVIDIA/nvidia-oci-samples/issues) for bug reports or feature requests
- Refer to each sample's README for sample-specific guidance

To report a security vulnerability, follow the process in [SECURITY.md](SECURITY.md).

## License

See [LICENSE](LICENSE). This project is licensed under the Apache License 2.0.

## Maintainers

See [MAINTAINERS.md](MAINTAINERS.md).
