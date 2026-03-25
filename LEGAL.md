# Legal Basis

## Reverse Engineering for Interoperability

This project uses Apple's private `AppleNeuralEngine.framework` APIs discovered through runtime introspection (`NSClassFromString`, `objc_copyClassList`, `class_copyMethodList`). No Apple proprietary source code or binaries are included in this repository.

### Legal Framework

**DMCA Section 1201(f) — Reverse Engineering Exception:**
> A person may reverse engineer ... for the sole purpose of identifying and analyzing those elements of the program that are necessary to achieve interoperability of an independently created computer program with other programs.

This project reverse engineers the ANE interface to achieve interoperability — enabling independently created training software to run on Apple's Neural Engine hardware.

**Sega v. Accolade, 977 F.2d 1510 (9th Cir. 1992):**
The court ruled that reverse engineering for interoperability constitutes fair use, even when it involves copying copyrighted code during the analysis process. The final product (this repository) contains no Apple code.

### What This Project Does NOT Do

- Does NOT include any Apple source code, binaries, or proprietary data
- Does NOT circumvent copy protection or DRM
- Does NOT bypass security mechanisms
- Does NOT distribute Apple's private frameworks

### What This Project DOES

- Calls documented Objective-C runtime APIs (`objc_msgSend`, `NSClassFromString`)
- Discovers class names and method signatures via public runtime introspection
- Generates MIL (Machine Learning Intermediate Language) programs as text
- Passes generated programs to the ANE compiler via discovered APIs
- All code is original, independently written

### Apple's Public ANE Training APIs (Historical Context)

Apple previously provided public APIs for ANE training:

| Framework | API | Status |
|-----------|-----|--------|
| **MLCompute** (iOS 14+) | `MLCDevice.ane()` | **Deprecated** |
| **MLCompute** (iOS 14+) | `MLCTrainingGraph` | **Deprecated** |
| **CoreML** | `MLUpdateTask` | Active (last layers only) |

Apple deprecated MLCompute without providing a replacement for ANE training. CoreML's on-device training (`MLUpdateTask`) only supports fine-tuning the last fully-connected and convolution layers — insufficient for full transformer training.

This project fills the gap left by Apple's deprecation of MLCompute's ANE training capabilities.

### Precedent

The [maderix/ANE](https://github.com/maderix/ANE) project uses the same approach (MIT licensed, published March 2026). Apple has not taken legal action against that project or any similar reverse engineering efforts targeting the Neural Engine.

### Distribution Restrictions

Due to private API usage, this software **cannot** be distributed via the Apple App Store. Valid distribution channels:
- Source code (GitHub)
- TestFlight (limited review)
- Ad-hoc signing (developer devices)
- Enterprise distribution

### License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
