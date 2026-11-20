# RailVision AI — Visão Computacional e Medição Visual

**Documento fonte:** Especificação Unificada v1.2  
**Última atualização:** 2026-11-19  

## 1. Câmera e Perspectiva

- **Modalidade:** exclusivamente câmera **RGB convencional** (sem LiDAR, laser, térmica ou sensor de profundidade).
- **Perspectiva preferencial:** elevada e oblíqua (*elevated oblique*), permitindo visualizar o interior dos vagões *open-top*, as bordas superiores e o perfil da carga.
- **Triagem obrigatória (Fase 2):** como o projeto utiliza apenas fontes públicas licenciadas, cada vídeo será classificado em:
  - `elevated_oblique_cargo_visible`
  - `top_down_cargo_visible`
  - `side_or_ground_level_cargo_not_visible` (útil somente para detecção/tracking, nunca para material ou carregamento)
  - `other_or_unusable`

## 2. Pipeline Proposto (Fases 3–6)

1. Validação do arquivo de vídeo e extração de metadados reais (`duration_ms`, `resolution`, `fps`, `codec`).
2. Amostragem de frames preservando `frame_index` e `frame_timestamp_ms`, evitando redundância excessiva.
3. Detecção de vagões abertos (*open-top*).
4. Rastreamento entre frames (`track_id` temporário dentro da inspeção).
5. Seleção de observações de qualidade (enquadramento, nitidez, baixa oclusão).
6. Classificação de material (comparar experimentalmente recorte do vagão vs. recorte da região de carga).
7. Estimativa ordinal do nível de carregamento.
8. Consolidação temporal por `track_id` (votação ponderada para material, mediana/moda para nível ordinal, registro de discordâncias e `review_required`).
9. Persistência de resultados e evidências para visualização na UI.

## 3. Seleção de Modelos e Política de Experimentos

- **Detecção (baseline):** `YOLO26n` (adequado para partida na RTX 4060 Ti 8 GB); comparar com `YOLO26s` ou `RF-DETR` somente diante de limitação identificada e hipótese documentada.
- **Rastreamento:** `ByteTrack` e `BoT-SORT` avaliados em vídeos representativos (atenção a oclusões, vagões visualmente idênticos em sequência e parada/retomada).
- **Identificação de material:**
  - Classes iniciais provisórias: `iron_ore`, `coal`, `limestone`, `sand`, `other_material` (+ rejeição `unknown` e `material = null` quando `loading_level = empty`).
  - Baseline candidato: `YOLO26-cls`; `ConvNeXt-Tiny` / `EfficientNetV2-S` apenas se dados justificarem comparação.
- **Estimativa de nível de carregamento:**
  - **Alvo MVP:** categorias ordinais `empty`, `low`, `medium`, `high`, `very_high`. Percentual numérico e volume físico estão **deferidos** (`estimated_fill_percentage = null`).
  - **Abordagem A (primeiro baseline):** classificação ordinal direta no recorte (menor custo de anotação, sem premissa geométrica frágil).
  - **Abordagem B (condicional):** segmentação semântica (`YOLO26-sem` / `SegFormer`) + geometria/perspectiva somente se a auditoria de ângulos justificar o custo das máscaras por pixel e superar a Abordagem A em teste independente.

### Fatos verificados sobre YOLO26 (fonte: docs.ultralytics.com, verificado em 2026-11-19)
- Famílias oficiais: `yolo26{n,s,m,l,x}.pt`, `-cls.pt`, `-seg.pt`, `-sem.pt`, `-pose`, `-obb`, `-depth`.
- Exportação ONNX documentada pela Ultralytics para detecção e variantes `-sem`.
- **Hardware verificado (Fases 3–M5):** execução padrão via `pytorch_cuda` na NVIDIA GeForce RTX 4060 Ti 8 GB com paridade numérica validada (`minIoU >= 0.999`) em relação ao `onnx_cpu` e referência da Fase 4.

## 4. Métricas Planejadas

- **Detecção:** Precision, Recall, `mAP50`, `mAP50-95`, análise FP/FN.
- **Tracking:** `IDF1`, `MOTA`, consistência de identidade e acurácia de contagem por composição (exige subconjunto com IDs anotados).
- **Classificação de material:** Accuracy, Precision/Recall por classe, Macro F1, matriz de confusão e curva cobertura vs. acurácia com rejeição (`unknown`).
- **Carregamento ordinal:** matriz de confusão, acurácia com tolerância de 1 classe adjacente, *Quadratic Weighted Kappa* (QWK) e MAE sobre índice ordinal.
- **Calibração e sistema:** análise de incerteza/ECE, latência (ms), FPS, consumo de CPU/GPU e RAM/VRAM.
- **Limiares de aceite:** `null` até medição da baseline e aprovação do usuário.
