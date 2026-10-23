# Third-party sources and distribution boundaries

| Component | Pinned source | Included in source Git history? |
|---|---|---|
| NavRL feature architecture / optional initialization | Zhefan-Xu/NavRL, `3725bcc2e7c1be4ecf1455d922299ae85042603a` | Adapted architecture code with its MIT notice; original checkpoint downloaded separately |
| Go2 model and visual meshes | google-deepmind/mujoco_menagerie, `cda5236f98383f6fbaa4820c6b18dba892cd087f`, `unitree_go2/` | No; downloaded with upstream LICENSE and README |
| Frozen Go2 locomotion models | 11chens/SEA-Nav-Code, `560de73c094a750a6a358bc1920ebc9ef1d9f3c5`, `training/legged_gym/legged_gym/ctrl_model/` | No; downloaded separately |
| Adapted high-level navigation checkpoint | This project's Go2 training run | Yes; exported tensor-only state with an allowlist of public configuration metadata |

Source links: [NavRL](https://github.com/Zhefan-Xu/NavRL), [Menagerie Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/unitree_go2), [SEA-Nav](https://github.com/11chens/SEA-Nav-Code).

The MIT license at the repository root covers original adaptation code and documentation. It does not grant additional rights to third-party assets or model weights. NavRL-derived code is accompanied by `docs/NavRL-LICENSE.txt`. The public availability of a third-party model is not a blanket redistribution license. Review its upstream terms before use or redistribution. This repository links the exact public files instead of republishing them.

No robot-vendor deployment SDK, real-robot control transport or external deployment repository is bundled. The locomotion interface is implemented against the released TorchScript signatures and published joint configuration. It is a MuJoCo adaptation rather than a claim of author-endorsed deployment.
