# BIPA — dataset, inferência e avaliação do modelo publicado

O projeto usa o [dataset BIPA](https://huggingface.co/datasets/thiagomonteles/BIPA) e o [modelo BIPA já treinado](https://huggingface.co/thiagomonteles/BIPA_g2p_Multidialect_Byt5). Ele não treina o modelo.

O código do dataset fica separado em `bipa_dataset.py`. Esse módulo pode baixar, limpar, validar, resumir e particionar os dados sem carregar o modelo. `bipa.py` importa o mesmo módulo quando precisa dos dados para fazer inferência ou calcular as métricas.

## 1. Instalar as dependências

No terminal, na pasta do projeto:

```powershell
python -m pip install -r requirements.txt
```

Na primeira execução, é necessária uma conexão com a internet para baixar o dataset e, ao usar o modelo, o checkpoint. Os arquivos baixados ficam no cache do Hugging Face.

## 2. Analisar e verificar o dataset

Para baixar a revisão fixada do dataset, executar as verificações e mostrar um resumo:

```powershell
python bipa_dataset.py
```

O relatório mostra o número de linhas de origem, linhas vazias e duplicatas removidas, registros válidos, pares dialeto-palavra, tamanhos de palavras, quantidade de referências, distribuição por dialeto e tamanhos das partições. O script também verifica as colunas obrigatórias, valores vazios, dialetos desconhecidos, duplicatas remanescentes e vazamento de pares entre as partições. Se uma verificação falhar, o programa informa o erro e termina.

Para inspecionar um TSV local com as colunas `dialeto`, `palavra` e `pronuncia`:

```powershell
python bipa_dataset.py --input "dados\meu_dataset.tsv"
```

Para salvar os dados já limpos em outro TSV, que também pode ser usado novamente com `--input`:

```powershell
python bipa_dataset.py --save-cleaned "dados\bipa_limpo.tsv"
```

Também é possível combinar `--input` e `--save-cleaned`. O arquivo salvo mantém as colunas `dialeto`, `palavra` e `pronuncia`, com os rótulos de dialeto normalizados.

As verificações principais também podem ser chamadas em outro código Python:

```python
from bipa_dataset import create_splits, load_dataset

dataset = load_dataset()
splits = create_splits(dataset)

assert not dataset[["word", "pronunciation"]].apply(
    lambda column: column.str.strip().eq("").any()
).any()
assert not dataset.duplicated(["dialect", "word", "pronunciation"]).any()

pair_sets = {
    name: set(zip(part["dialect"], part["word"]))
    for name, part in splits.items()
}
assert pair_sets["train"].isdisjoint(pair_sets["validation"])
assert pair_sets["train"].isdisjoint(pair_sets["test"])
assert pair_sets["validation"].isdisjoint(pair_sets["test"])
```

As partições são agrupadas por dialeto e palavra e estratificadas por dialeto e comprimento da palavra. As proporções visam 75% treino, 15% validação e 10% teste. A semente fixa torna a partição reproduzível para as revisões fixadas. Os índices originais do artigo não foram publicados, portanto os valores podem diferir.

## 3. Gerar pronúncias com o modelo

Para uma palavra:

```powershell
python bipa.py --dialect "[brasil]" --text casa
```

Para várias palavras:

```powershell
python bipa.py --dialect "[são-paulo]" --text carro noite casa
```

Dialetos aceitos: `[brasil]`, `[rio-de-janeiro]`, `[são-paulo]`, `[regiao-sul]`, `[regiao-nordeste]` e `[regiao-centro-oeste]`.

## 4. Avaliar o modelo com Minimum PER

Avaliação completa no conjunto de teste e em todos os dialetos:

```powershell
python bipa.py --evaluate
```

Para comparar mais diretamente com a Tabela 4 do artigo, que avalia o dialeto geral brasileiro no teste:

```powershell
python bipa.py --evaluate --dialect "[brasil]"
```

Para avaliar a partição de validação:

```powershell
python bipa.py --evaluate --split validation
```

Uma amostra pequena para testar o fluxo, não para usar como métrica final:

```powershell
python bipa.py --evaluate --limit 20
```

A avaliação completa pode demorar, especialmente em CPU. O programa usa GPU NVIDIA quando disponível e CPU como alternativa. O carregamento e as verificações do dataset são feitos por `bipa_dataset.py`; `bipa.py` carrega o checkpoint, gera as pronúncias e calcula as métricas.

Para cada par dialeto-palavra da partição selecionada, o modelo gera uma pronúncia. A previsão é comparada com as referências daquele par, usando a referência de menor PER, calculada pela distância de Levenshtein normalizada. A saída mantém Minimum PER macro, micro e por dialeto. Para o dialeto `[brasil]` no teste, o artigo reporta 6,07%; não se espera necessariamente o mesmo valor devido a possíveis diferenças nas partições e nos detalhes de segmentação.

As métricas e previsões são salvas em `runs/evaluation/`. `metrics.json` registra a partição, as revisões do modelo e dataset, a semente, a quantidade de pares e os resultados; `predictions.tsv` contém os detalhes por palavra.

## Arquivos e licenças

- `bipa_dataset.py`: leitura, limpeza, validação, análise e partições do dataset.
- `bipa.py`: inferência e avaliação do modelo publicado.
- `requirements.txt`: dependências do projeto.
- `2026.propor-1.47.pdf`: artigo do PROPOR 2026.
- `LICENSE`: licença do código deste repositório.

O dataset BIPA indica licença CC BY-SA 4.0; o modelo indica Apache-2.0. O PDF e esses materiais têm termos próprios. Consulte as páginas correspondentes antes de redistribuí-los e cite o artigo ao compartilhar resultados.
