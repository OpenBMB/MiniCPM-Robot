# Third-party notices for simulation evaluation

Files under `evaluation/` and `deployment/model_server/tools/` include code
adapted from
[starVLA commit `631aae02`](https://github.com/starVLA/starVLA/tree/631aae02afe6d95876e923ff518e8ff2ab9a2f88).

Migrated or adapted material includes:

- LIBERO and CALVIN evaluation loops and launch structure.
- RoboTwin policy interface and multi-GPU launch structure.
- `common/adaptive_ensemble.py`.
- `calvin/eval_sequences.json`.
- The WebSocket client/server and NumPy MessagePack transport under
  `deployment/model_server/tools/`.

The RMBench closed-loop rollout, continuous dense executor, per-camera history
buffer, and memVLA server were ported from OpenBMB's memVLA/RMBench evaluation
code. They are OpenBMB code under the repository's Apache-2.0 license, not part
of the starVLA-derived material above.

Modified starVLA-derived source files retain the starVLA copyright header and
identify OpenBMB modifications.

## starVLA upstream LICENSE text

The upstream LICENSE at the pinned commit includes repository-history guidance
in addition to the MIT terms; the complete upstream text is reproduced below.

```text
MIT License

Copyright (c) StarVLA Team.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Rebases are allowed for forks and feature branches. When rebasing from
upstream StarVLA, use descriptive commit messages, e.g.,
"chore: clone from StarVLA".
Preserve attribution: keep at least the two latest upstream StarVLA commits as
separate.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

This repository records the exact source commit above because the files were
migrated as modified source rather than by rebasing starVLA history.

## External simulator projects

The following projects are runtime dependencies and aren't vendored:

- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- CALVIN: <https://github.com/mees/calvin>
- RoboFlamingo evaluation utilities:
  <https://github.com/RoboFlamingo/RoboFlamingo>
- RoboTwin: <https://github.com/RoboTwin-Platform/RoboTwin>
- RMBench (RoboTwin 2.0 / SAPIEN):
  <https://github.com/RoboTwin-Platform/RMBench>; simulation assets:
  ModelScope dataset `keithyc/RMBench_sim`.

Their source code, datasets, assets, and generated outputs remain subject to
their respective licenses and citation requirements.

## RoboFlamingo license

The migrated CALVIN evaluation flow follows RoboFlamingo's evaluation protocol.

```text
MIT License

Copyright (c) 2023 XinghangLi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## msgpack-numpy license

The NumPy MessagePack implementation in
`deployment/model_server/tools/msgpack_numpy.py` is adapted through starVLA
from [`msgpack-numpy`](https://github.com/lebedov/msgpack-numpy). The
implementation rejects object arrays rather than falling back to pickle.

```text
Copyright (c) 2013-2022, Lev E. Givon.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Lev E. Givon nor the names of any contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
