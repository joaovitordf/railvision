# RailVision AI — Estratégia de Dados, Proveniência e Splits

**Documento fonte:** Especificação Unificada v1.2  
**Última atualização:** 2026-11-19  

## 1. Estratégia Geral

1. Auditar primeiro datasets públicos candidatos para detecção de trens/vagões (`TWIST`) e aparência auxiliar (`UFOP Iron Ore Conveyor`), baixando e inspecionando arquivos reais apenas após confirmar licença compatível.
2. Para classificação de material e nível ordinal de carregamento em vagões abertos, construir um **conjunto customizado** a partir de fontes públicas com licença explícita e compatível.
3. Nunca misturar domínios (ex.: minério em correia transportadora vs. vagão ferroviário) sem medir impacto em teste independente.

## 2. Regras de Coleta de Vídeos Públicos

Disponibilidade pública na internet **não** equivale a autorização para download, armazenamento ou treinamento de modelos. Antes de baixar qualquer vídeo:
1. Identificar autor/titular, URL original, licença explícita e termos da plataforma.
2. Confirmar que a licença permite armazenamento local, análise e treinamento/avaliação (ex.: CC0/domínio público, CC BY ou permissão escrita).
3. Não utilizar conteúdo `NC`/`ND` ou com termos ambíguos para treinamento sem análise específica e autorização.
4. Registrar um manifesto com os campos: `id`, `source_url`, `retrieved_at`, `author_or_owner`, `license`, `license_url`, `permitted_uses`, `sha256`, `duration_s`, `resolution`, `fps`, `angle_category`, `cargo_visible`, `wagon_type`, `notes`.

## 3. Regra de Admissão de Classes e Anotação

- **Admissão de classe:** nenhuma classe provisória (`iron_ore`, `coal`, `limestone`, `sand`, `other_material`) entra no treinamento sem amostras suficientes e representativas de **múltiplos vídeos/sessões distintos**. Limiares mínimos serão definidos após a auditoria (Fase 2).
- **Confiabilidade do rótulo (`label_basis`):** registrar sempre `visual_only`, `source_metadata` ou `expert_confirmed`.
- **Guia de anotação:** definir fronteiras operacionais de *visual fill* antes de iniciar anotação manual em escala. Toda anotação assistida por IA exige revisão humana.

## 4. Política de Split (Sem Vazamento)

- **Obrigatório:** split estritamente **por grupo (vídeo / sessão de gravação / fonte)**.
- É proibido distribuir frames adjacentes ou quase-duplicados do mesmo vídeo entre treino, validação e teste.
- Manter conjunto de teste final independente (não usado em seleção de hiperparâmetros).

## 5. Dados Sintéticos

Permitidos apenas após identificar lacuna real nos dados auditados, devidamente separados dos dados reais; avaliação final válida apenas em amostras reais independentes.

## 6. Higiene de Repositório

- Vídeos originais, datasets, frames extraídos, máscaras, pesos (`.pt`, `.pth`, `.onnx`, `.ckpt`) e arquivos de upload ficam em `data/` e `artifacts/`, **ignorados pelo Git**.
- Versionam-se no Git apenas código, scripts, configurações, `specs/` e manifestos textuais (proveniência, licenças, hashes SHA-256).
