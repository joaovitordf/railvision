# RailVision AI — Inventário de Licenças

**Última verificação:** 2026-11-20  
**Finalidade do projeto:** Portfólio e estudo (revisar obrigações de atribuição e ShareAlike antes de eventual publicação pública ou uso comercial).

## 1. Frameworks e Dependências Principais

| Componente | Uso | Licença | Observações |
|---|---|---|---|
| FastAPI (`0.165.0`) + Uvicorn (`0.54.0`) | Backend Web/REST unificado (`backend/`) | MIT / BSD-3-Clause | Compatível |
| SQLAlchemy (`2.1.9`) + Alembic (`1.20.1`) + `psycopg[binary]` (`3.3.9`) | Persistência ORM e migrations PostgreSQL | MIT / LGPLv3+ (`psycopg`) | Compatível |
| PyAV (`av==19.0.2`) + `libx264` | Decodificação e gravação H.264/MP4 controlada | BSD-3-Clause (`av`) / LGPL-GPL conforme build FFmpeg | Verificar pacote FFmpeg caso haja redistribuição comercial externa |
| PostgreSQL (`17.12-alpine` Docker) | Banco de dados relacional | PostgreSQL License | Imagem fixada por versão |
| Angular / Angular Material (`22.3.0`) | Frontend web | MIT | Compatível |
| PyTorch (`2.9.1+cu128`) + TorchVision (`0.24.1+cu128`) | Runtime CUDA padrão de inferência e treino (`ml/.venv`) | BSD-3-Clause | Compatível |
| Python ONNX Tooling (`onnx==1.23.2`, `onnxruntime==1.32.0`) | Exportação, paridade e runtime CPU selecionável | Apache-2.0 / MIT | Compatível |
| Ultralytics YOLO26 (`8.4.215`) + `lap` ByteTrack | Detecção e rastreamento (treino offline e inferência local) | **AGPL-3.0 / Enterprise** | Verificado em 2026-11-19 (`docs.ultralytics.com` / GitHub). |
| RF-DETR *(alternativa condicional)* | Detecção alternativa | **Apache-2.0** (Nano/Small/Medium/Large/Seg) ou **Roboflow PML 1.0** (`rfdetr[plus]`) | Verificado em 2026-11-19 (`github.com/roboflow/rf-detr`) |

## 2. Datasets Auditados

| Fonte | URL | Licença verificada | Regra no RailVision AI |
|---|---|---|---|
| TWIST | <https://zenodo.org/records/19472084> | `CC-BY-4.0` | Permitido com atribuição (`ds-kiel/TWIST`); usar após deduplicação e remapeamento de classes |
| UFOP Iron Ore Conveyor | <https://data.mendeley.com/datasets/s25x2bnshz/1> | `CC BY 4.0` | Permitido com atribuição; fora do domínio ferroviário (correia transportadora / `nc=1 conveyor`) |
| Wagons Images Classification | <https://huggingface.co/datasets/UniqueData/wagons-images-classification> | `CC-BY-NC-ND-4.0` | **Não usar para treino ou derivação**; apenas referência visual |

## 3. Vídeos Públicos Triados (Wikimedia Commons — Verificados em 2026-11-20)

| ID local | Autor / Titular | URL | Licença | Obrigação |
|---|---|---|---|---|
| `vid_01` | Maximilian Schönherr | <https://commons.wikimedia.org/wiki/File:Langer_unbeladener_G%C3%BCterzug,_von_oben_%E2%80%93_K%C3%B6ln_2024.webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike |
| `vid_02` | Erik Pålsson | <https://commons.wikimedia.org/wiki/File:Soundscapes_of_Norrk%C3%B6ping_(9)_-_Bridge.webm> | `CC BY 3.0` | Atribuição (não utilizável para trens) |
| `vid_03` | Motopäevik | <https://commons.wikimedia.org/wiki/File:Stadler_FLIRT_EMU_1323_and_1311_at_Vesse_station_from_above.webm> | `CC BY 3.0` | Atribuição (trem de passageiros) |
| `vid_04` | rustam79 | <https://commons.wikimedia.org/wiki/File:2ES10-003_with_freight_train_from_above.webm> | `CC BY 3.0` | Atribuição |
| `vid_05` | ChristianSchd | <https://commons.wikimedia.org/wiki/File:Timelapse_freight_bypass_railway_Eichenbrink_Davenstedt_Hannover_Germany.webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike |
| `vid_06` | Steven Pavlov | <https://commons.wikimedia.org/wiki/File:Seattle_%E2%80%94_Freight_train_moves_across_the_Salmon_Bay_Bridge.webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike |
| `vid_07` | 掬茶 | <https://commons.wikimedia.org/wiki/File:Freight_train_crossing_the_Kyu-Omono_River_Bridge_20210301.webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike |
| `vid_08` | Rainerhaufe | <https://commons.wikimedia.org/wiki/File:Dampfloks_auf_der_Rampe_Tharandt-Klingenberg.webm> | `CC BY-SA 3.0` | Atribuição + ShareAlike |
| `vid_09` | Cjp24 | <https://commons.wikimedia.org/wiki/File:Vossloh_G_1206_BB_of_VFLI_(video).webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike |
| `vid_10` | Rmv | <https://commons.wikimedia.org/wiki/File:Fluorescent-lamp-electronic-ballast.webm> | `CC BY-SA 3.0` | Atribuição + ShareAlike (falso positivo — lâmpada, descartado) |
| `vid_11` | The Hellenic Trainspotters | <https://commons.wikimedia.org/wiki/File:Special_Transfer_of_K2_Ballast_Cleaner_to_Kiato..webm> | `CC BY 3.0` | Atribuição (equipamento noturno, descartado) |
| `vid_12` | Bote Rail | <https://commons.wikimedia.org/wiki/File:F.E.C._A.A.F._Ballast_Tamper-Regulator.webm> | `CC BY 3.0` | Atribuição (máquina de via, descartado) |
| `vid_13` | Leppus | <https://commons.wikimedia.org/wiki/File:Bahn_bei_Roetha.webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike (oclusão lateral, descartado) |
| `vid_14` | RAILWAY BAY | <https://commons.wikimedia.org/wiki/File:Coal_Train_Locomotive_Exit_The_Tunnel_Crested_And_Passed_Skyline.webm> | `CC BY 3.0` | Atribuição (candidato futuro Fase 5 — carvão) |
| `vid_15` | Discott | <https://commons.wikimedia.org/wiki/File:Sishen%E2%80%93Saldanha_railway_line.ogv> | `CC BY-SA 3.0` | Atribuição + ShareAlike (vagões abertos minério de ferro; oclusão parcial de superfície) |
| `vid_16` | Cjp24 | <https://commons.wikimedia.org/wiki/File:Vossloh_G_1206_BB_61735_and_BB_61744,_VFLI_-_signal_D_(video).webm> | `CC BY-SA 4.0` | Atribuição + ShareAlike (candidato futuro Fase 5 — hopper com lastro/brita, mesma campanha de `vid_09`) |

*(Observação sobre `CC BY-SA`: se derivar datasets/pesos públicos sob `ShareAlike`, revisar se a distribuição futura do artefato derivado exige a mesma licença compatível.)*
