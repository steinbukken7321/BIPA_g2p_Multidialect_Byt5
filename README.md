# Reprodução do experimento BIPA / ByT5

Este projeto prepara o dataset BIPA, faz fine-tuning de `google/byt5-small` e
avalia o checkpoint com a métrica Minimum Phoneme Error Rate (Minimum PER)
descrita no artigo:

> Thiago Monteles de Sousa, Lucas Rafael Gris, and Nádia Félix Felipe da Silva.
> 2026. *BIPA: Brazilian Portuguese Phonetic Dataset with Dialectal Variations
> in IPA Standard*. PROPOR 2026, pp. 478–487.
> https://aclanthology.org/2026.propor-1.47/

Dataset: https://huggingface.co/datasets/thiagomonteles/BIPA  
Modelo publicado: https://huggingface.co/thiagomonteles/BIPA_g2p_Multidialect_Byt5

## Ambiente

Use Python 3.11 ou 3.12. O fine-tuning do ByT5-small é pesado; recomenda-se uma
GPU NVIDIA com memória suficiente. O batch size por dispositivo é 96, como
descrito no artigo, e pode precisar ser reduzido se houver falta de memória.
Reduzi-lo altera a configuração do experimento.

No PowerShell, a partir da pasta do projeto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Instale uma versão de PyTorch compatível com seu sistema/CUDA se a instalação
pelo arquivo de dependências não configurar o backend adequado:
https://pytorch.org/get-started/locally/

## 1. Preparar o BIPA

```powershell
python reproduce_bipa.py prepare
```

O script baixa o TSV público do Hugging Face, mapeia os nomes dos dialetos para
as etiquetas usadas pelo modelo, remove duplicatas exatas de
`(dialeto, palavra, pronúncia)` e gera:

- `data/processed/train.tsv`
- `data/processed/validation.tsv`
- `data/processed/test.tsv`
- `data/processed/split_manifest.json`

As partições são feitas sobre pares únicos `(dialeto, palavra)`, nunca separando
pronúncias da mesma palavra/dialeto entre treino, validação e teste. A
estratificação combina etiqueta de dialeto e comprimento exato da palavra em
caracteres. Sem `--seed`, o script gera uma seed aleatória e grava o valor usado
em `data/processed/split_manifest.json`. Para repetir a mesma divisão, informe
esse valor em `--seed`.
O script fixa a revisão do dataset `d1e81df1565756229bc20d6841716a7a5560fc80`
e a revisão do modelo base `google/byt5-small`
(`68377bdc18a2ffec8a0533fef03b1c513a4dd49d`) para que as entradas não mudem
silenciosamente.

## 2. Retreinar desde o modelo base

```powershell
python reproduce_bipa.py train --seed 42
```

Execute `prepare` ao menos uma vez e confirme que os três arquivos em
`data/processed/` foram criados antes de iniciar o treinamento. A seed `42` aqui
controla a aleatoriedade do treino; a seed usada para dividir os dados é a
registrada no manifesto por `prepare`.

`reproduce_bipa.py` prepara os dados e chama a implementação do modelo em
[`bipa_model.py`](./bipa_model.py). O treinamento parte de `google/byt5-small`
(não do checkpoint já ajustado) e usa 10 épocas, learning rate `1e-4`,
Adafactor, weight decay `0.05`, label smoothing `0.1`, warm-up de 10%,
scheduler linear, beam search de largura 6 e saída máxima de 35 tokens. A
seleção do melhor checkpoint usa o Minimum PER na validação. O warm-up de 10%
é convertido em número de passos a partir do tamanho do treino e do batch size,
para compatibilidade com versões recentes do Transformers.

Resultados e pesos são gravados em `runs/bipa_byt5/`, incluindo:

- `best-model/`: melhor checkpoint de validação;
- `training_history.json`: métricas registradas durante o treino;
- `validation/metrics.json` e `validation/predictions.tsv`;
- `test/metrics.json` e `test/predictions.tsv`.

O resultado final de teste deve ser lido em `runs/bipa_byt5/test/metrics.json`.
O teste só é usado após seleção do checkpoint pela validação.

## 3. Inferência

O comando `inference` pode gerar transcrições para palavras ou avaliar o modelo
em um split preparado. Para inferência em uma palavra:

```powershell
python reproduce_bipa.py inference --dialect "[são-paulo]" --text carro
```

Mais de uma palavra pode ser passada em `--text`:

```powershell
python reproduce_bipa.py inference --dialect "[brasil]" --text casa noite carro
```

Por padrão, a inferência usa o checkpoint publicado, fixado na revisão
`9068d5499496440873884c1e9ab91adaf971fdda`. Para usar o checkpoint treinado
localmente:

```powershell
python reproduce_bipa.py inference `
  --model runs/bipa_byt5/best-model `
  --dialect "[são-paulo]" `
  --text carro
```

Para calcular Minimum PER no teste usando o modelo publicado:

```powershell
python reproduce_bipa.py inference --split test
```

Troque `--split test` por `--split validation` para avaliar a validação. Para
inferência com uma revisão publicada mais recente, passe `--revision main`.

## Métrica

Para cada par `(dialeto, palavra)`, o modelo gera uma pronúncia e ela é
comparada com todas as transcrições distintas disponíveis para aquele par. O
Minimum PER escolhe a referência com menor distância de edição normalizada.
Cada unidade é um grafema Unicode estendido (`\X`), para preservar diacríticos
combinantes do IPA. O relatório inclui:

- `min_per`: média macro do Minimum PER por par dialeto-palavra;
- `micro_per`: soma das menores distâncias dividida pelo total de símbolos nas
  referências escolhidas;
- `exact_match_percent`: porcentagem de pares com transcrição idêntica;
- as mesmas métricas separadas por dialeto.

Os resultados podem diferir de PER publicado se a implementação original usou
outra agregação de PER ou outra segmentação de símbolos; o artigo não especifica
todos esses detalhes.

## Limites para a reprodução exata

O procedimento segue os hiperparâmetros e o tamanho das partições reportados,
mas não é possível garantir os mesmos números do artigo apenas com os arquivos
públicos:

1. Os autores não publicaram a seed nem os índices dos exemplos de cada
   partição. O `42` deste projeto é uma escolha explícita para repetibilidade,
   não uma recuperação da divisão original.
2. Há uma inconsistência no texto: os tamanhos publicados
   (`219.921 / 43.699 / 29.242`) correspondem aproximadamente a
   `75% / 15% / 10%`, enquanto a descrição da validação pode ser lida como 10%
   do total. Este script segue as proporções dos tamanhos publicados.
3. Software, versão do modelo-base, hardware e detalhes não informados da
   implementação original também influenciam os resultados.
4. O artigo reporta 3,6% como melhor resultado de validação. A tabela por
   dialeto lista 6,07% para “Brasil (geral)” e a comparação no teste também
   mostra 6,07%; portanto, compare separadamente validação, teste e dialetos,
   em vez de tratar 3,6% como valor esperado para qualquer execução.

O manifesto salva o hash SHA-256 do TSV baixado, a seed efetivamente usada,
método e contagens das partições para que a execução possa ser documentada e
repetida. A seed de `train` controla a aleatoriedade do treinamento e é
independente da seed usada para criar as partições.

## GitHub e mudança de máquina

O `.gitignore` exclui ambientes virtuais, os TSVs processados e os resultados de
treinamento em `runs/`, que podem ser recriados. O manifesto da divisão permanece
versionado; na nova máquina, use a seed registrada nele para refazer a mesma
partição:

```powershell
python reproduce_bipa.py prepare --seed <seed-do-manifesto>
```

Os checkpoints não são enviados ao GitHub. Se precisar preservar o resultado de
um treinamento já iniciado, transfira a pasta `runs/bipa_byt5/` separadamente
(por exemplo, por um disco externo ou armazenamento de arquivos). O comando
`train` atual não retoma automaticamente de um checkpoint; sem implementar essa
retomada, será necessário iniciar o treinamento novamente do modelo base.

## Licença e citação

O código original deste projeto está sob a licença MIT; veja [LICENSE](LICENSE).
Essa licença não substitui nem altera os termos dos materiais de terceiros:

- O dataset [BIPA](https://huggingface.co/datasets/thiagomonteles/BIPA) é
  publicado sob CC BY-SA 4.0 e deriva do Wiktionary. Respeite atribuição e
  compartilhamento pela mesma licença ao redistribuir material do dataset.
- O modelo ajustado
  [BIPA_g2p_Multidialect_Byt5](https://huggingface.co/thiagomonteles/BIPA_g2p_Multidialect_Byt5)
  e o modelo base [google/byt5-small](https://huggingface.co/google/byt5-small)
  indicam Apache-2.0 no Hugging Face.
- O artigo PDF incluído neste repositório está disponível pela ACL Anthology
  sob CC BY 4.0, conforme a
  [política de copyright da ACL Anthology](https://aclanthology.org/faq/copyright/).

O script baixa o dataset; os arquivos gerados localmente são ignorados pelo
Git. Ao compartilhar resultados, cite o artigo PROPOR 2026 acima.
