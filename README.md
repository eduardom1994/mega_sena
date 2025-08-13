# 🎲 Analisador e Previsor da Mega Sena

Este programa coleta todos os dados históricos da Mega Sena da internet e realiza análises estatísticas para gerar previsões para o próximo sorteio.

## 📋 Funcionalidades

- **Coleta de Dados**: Obtém todos os resultados históricos da Mega Sena via API
- **Análise Estatística**: Analisa frequências, padrões e tendências dos números
- **Visualizações**: Gera gráficos e charts para análise visual
- **Previsões**: Gera múltiplas previsões baseadas em análise estatística
- **Relatórios**: Produz relatórios detalhados com insights

## 🚀 Como Usar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o Programa
```bash
python mega_sena_analyzer.py
```

## 📊 O que o Programa Faz

### Coleta de Dados
- Conecta-se à API das Loterias CAIXA
- Coleta todos os resultados históricos desde 1996
- Processa e limpa os dados coletados

### Análises Realizadas
- **Frequência dos Números**: Quais números saem mais/menos
- **Padrões Pares/Ímpares**: Distribuição entre números pares e ímpares
- **Análise de Soma**: Soma total dos números sorteados
- **Números Consecutivos**: Frequência de números em sequência
- **Amplitude**: Diferença entre maior e menor número

### Estratégias de Previsão
1. **Análise de Frequência**: Considera números "quentes" e "frios"
2. **Balanceamento**: Combina números frequentes com menos frequentes
3. **Padrões Históricos**: Usa tendências identificadas nos dados

## 📈 Saídas do Programa

- **mega_sena_data.csv**: Todos os dados coletados
- **mega_sena_analysis.png**: Gráficos de análise
- **Relatório na tela**: Estatísticas detalhadas e previsões

## ⚠️ Aviso Importante

Este programa é apenas para fins **educacionais e de entretenimento**. 

A Mega Sena é um jogo de azar e não há garantia de que análises estatísticas resultem em vitórias. Jogue com responsabilidade!

## 🛠️ Tecnologias Utilizadas

- **Python 3.7+**
- **Requests**: Para coleta de dados via API
- **Pandas**: Para manipulação de dados
- **NumPy**: Para cálculos estatísticos
- **Matplotlib/Seaborn**: Para visualizações
- **API Loterias CAIXA**: Fonte dos dados

## 📞 Suporte

Se encontrar problemas:
1. Verifique sua conexão com a internet
2. Certifique-se de que todas as dependências estão instaladas
3. A API pode estar temporariamente indisponível

---

**Desenvolvido para análise estatística da Mega Sena** 🍀
