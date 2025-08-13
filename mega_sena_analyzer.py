#!/usr/bin/env python3
"""
Mega Sena Data Analyzer and Predictor
=====================================

This program collects all historical Mega Sena lottery data from the internet
and performs statistical analysis to predict the next draw.

Features:
- Data collection from multiple sources
- Statistical analysis of number patterns
- Frequency analysis
- Prediction algorithms
- Data visualization
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import json
import time
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MegaSenaAnalyzer:
    def __init__(self):
        self.api_base_url = "https://loteriascaixa-api.herokuapp.com/api"
        self.data = []
        self.df = None
        
    def collect_all_data(self) -> bool:
        """
        Collect historical Mega Sena data from the API (incremental update)
        """
        print("🎲 Verificando dados históricos da Mega Sena...")
        
        try:
            # Check if we have existing data
            csv_file = "mega_sena_data.csv"
            last_contest = 0
            
            if os.path.exists(csv_file):
                try:
                    existing_df = pd.read_csv(csv_file)
                    if not existing_df.empty:
                        last_contest = existing_df['concurso'].max()
                        print(f"📁 Dados existentes encontrados até o concurso {last_contest}")
                        
                        # Load existing data
                        for _, row in existing_df.iterrows():
                            contest_data = {
                                'concurso': row['concurso'],
                                'data': row['data'],
                                'dezenas': [str(row[f'numero_{i}']) for i in range(1, 7)]
                            }
                            self.data.append(contest_data)
                        print(f"✅ Carregados {len(self.data)} concursos existentes")
                except Exception as e:
                    print(f"⚠️  Erro ao carregar dados existentes: {e}")
                    print("   Iniciando coleta completa...")
                    last_contest = 0
            
            # Get the latest contest number from API
            latest_url = f"{self.api_base_url}/megasena/latest"
            response = requests.get(latest_url, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Erro ao acessar API: {response.status_code}")
                return len(self.data) > 0  # Return True if we have existing data
                
            latest_data = response.json()
            latest_contest = latest_data.get('concurso', 0)
            
            print(f"📊 Último concurso disponível: {latest_contest}")
            
            # Determine range to collect
            start_contest = last_contest + 1 if last_contest > 0 else 1
            
            if start_contest > latest_contest:
                print("✅ Dados já estão atualizados!")
                return len(self.data) > 0
            
            print(f"⏳ Coletando concursos {start_contest} até {latest_contest}...")
            
            # Collect only new data with incremental save
            collected = 0
            errors = 0
            new_contests = []
            
            for contest_num in range(start_contest, latest_contest + 1):
                try:
                    contest_url = f"{self.api_base_url}/megasena/{contest_num}"
                    response = requests.get(contest_url, timeout=10)
                    
                    if response.status_code == 200:
                        contest_data = response.json()
                        self.data.append(contest_data)
                        new_contests.append(contest_data)
                        collected += 1
                        
                        # Save incrementally every 5 contests or at the end
                        if collected % 5 == 0 or contest_num == latest_contest:
                            self._save_incremental_data(new_contests, csv_file)
                            new_contests = []  # Clear the buffer
                            if collected % 50 == 0:  # Show progress every 50
                                print(f"✅ Coletados e salvos {collected} novos concursos...")
                    else:
                        errors += 1
                        
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    errors += 1
                    if errors > 20:  # Too many errors, stop
                        print(f"❌ Muitos erros na coleta. Parando...")
                        # Save what we have collected so far
                        if new_contests:
                            self._save_incremental_data(new_contests, csv_file)
                        break
                        
            # Save any remaining contests
            if new_contests:
                self._save_incremental_data(new_contests, csv_file)
                        
            if collected > 0:
                print(f"✅ Coleta finalizada! {collected} novos concursos coletados, {errors} erros")
            
            return len(self.data) > 0
            
        except Exception as e:
            print(f"❌ Erro na coleta de dados: {e}")
            return len(self.data) > 0  # Return True if we have existing data
    
    def process_data(self):
        """
        Process and clean the collected data
        """
        print("🔄 Processando dados coletados...")
        
        if not self.data:
            print("❌ Nenhum dado para processar!")
            return
            
        processed_data = []
        
        for contest in self.data:
            try:
                # Extract the drawn numbers
                numbers = contest.get('dezenas', [])
                if len(numbers) == 6:  # Mega Sena has 6 numbers
                    processed_data.append({
                        'concurso': contest.get('concurso'),
                        'data': contest.get('data'),
                        'numero_1': int(numbers[0]),
                        'numero_2': int(numbers[1]),
                        'numero_3': int(numbers[2]),
                        'numero_4': int(numbers[3]),
                        'numero_5': int(numbers[4]),
                        'numero_6': int(numbers[5]),
                        'acumulou': contest.get('acumulou', False),
                        'valor_acumulado': contest.get('valorAcumuladoProximoConcurso', 0)
                    })
            except (KeyError, ValueError, IndexError) as e:
                continue
                
        self.df = pd.DataFrame(processed_data)
        
        if not self.df.empty:
            # Convert date column
            self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
            self.df = self.df.sort_values('concurso').reset_index(drop=True)
            
            print(f"✅ {len(self.df)} concursos processados com sucesso!")
            print(f"📅 Período: {self.df['data'].min().strftime('%d/%m/%Y')} até {self.df['data'].max().strftime('%d/%m/%Y')}")
        else:
            print("❌ Erro no processamento dos dados!")
    
    def analyze_frequency(self) -> Dict:
        """
        Analyze number frequency patterns
        """
        print("📈 Analisando frequência dos números...")
        
        if self.df is None or self.df.empty:
            return {}
            
        # Get all drawn numbers
        all_numbers = []
        for col in ['numero_1', 'numero_2', 'numero_3', 'numero_4', 'numero_5', 'numero_6']:
            all_numbers.extend(self.df[col].tolist())
            
        # Count frequencies
        frequency_counter = Counter(all_numbers)
        
        # Calculate statistics
        total_draws = len(self.df)
        analysis = {
            'total_concursos': total_draws,
            'frequencias': dict(frequency_counter),
            'mais_sorteados': frequency_counter.most_common(10),
            'menos_sorteados': frequency_counter.most_common()[-10:],
            'media_aparicoes': np.mean(list(frequency_counter.values())),
            'numeros_nunca_sorteados': [i for i in range(1, 61) if i not in frequency_counter]
        }
        
        return analysis
    
    def analyze_patterns(self) -> Dict:
        """
        Analyze number patterns and sequences
        """
        print("🔍 Analisando padrões dos números...")
        
        if self.df is None or self.df.empty:
            return {}
            
        patterns = {
            'pares_impares': [],
            'sequencias': [],
            'soma_total': [],
            'amplitude': [],
            'numeros_consecutivos': []
        }
        
        for _, row in self.df.iterrows():
            numbers = sorted([row[f'numero_{i}'] for i in range(1, 7)])
            
            # Even/Odd analysis
            even_count = sum(1 for n in numbers if n % 2 == 0)
            patterns['pares_impares'].append(f"{even_count}P-{6-even_count}I")
            
            # Sum analysis
            patterns['soma_total'].append(sum(numbers))
            
            # Range analysis
            patterns['amplitude'].append(max(numbers) - min(numbers))
            
            # Consecutive numbers
            consecutive = 0
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive += 1
            patterns['numeros_consecutivos'].append(consecutive)
        
        # Calculate pattern statistics
        analysis = {
            'distribuicao_pares_impares': Counter(patterns['pares_impares']),
            'soma_media': np.mean(patterns['soma_total']),
            'soma_min': min(patterns['soma_total']),
            'soma_max': max(patterns['soma_total']),
            'amplitude_media': np.mean(patterns['amplitude']),
            'consecutivos_media': np.mean(patterns['numeros_consecutivos'])
        }
        
        return analysis
    
    def predict_next_draw(self) -> List[int]:
        """
        Generate prediction for next draw based on statistical analysis
        """
        print("🔮 Gerando previsão para o próximo sorteio...")
        
        if self.df is None or self.df.empty:
            return []
            
        # Get frequency analysis
        freq_analysis = self.analyze_frequency()
        pattern_analysis = self.analyze_patterns()
        
        # Strategy 1: Weighted random based on frequency
        frequencies = freq_analysis['frequencias']
        numbers = list(range(1, 61))
        weights = [frequencies.get(n, 0) + 1 for n in numbers]  # +1 to avoid zero weights
        
        # Strategy 2: Balance between hot and cold numbers
        hot_numbers = [n for n, _ in freq_analysis['mais_sorteados'][:20]]
        cold_numbers = [n for n in range(1, 61) if frequencies.get(n, 0) < freq_analysis['media_aparicoes']]
        
        # Generate prediction combining strategies
        prediction = []
        
        # Select 3-4 numbers from hot numbers
        hot_selection = np.random.choice(hot_numbers, size=min(4, len(hot_numbers)), replace=False)
        prediction.extend(hot_selection)
        
        # Select 2-3 numbers from cold numbers or medium frequency
        remaining_slots = 6 - len(prediction)
        available_numbers = [n for n in range(1, 61) if n not in prediction]
        
        if cold_numbers:
            cold_selection = np.random.choice(cold_numbers, size=min(remaining_slots, len(cold_numbers)), replace=False)
            prediction.extend(cold_selection)
        
        # Fill remaining slots randomly
        while len(prediction) < 6:
            available = [n for n in range(1, 61) if n not in prediction]
            if available:
                prediction.append(np.random.choice(available))
            else:
                break
                
        return sorted(prediction[:6])
    
    def generate_multiple_predictions(self, num_predictions: int = 5) -> List[List[int]]:
        """
        Generate multiple predictions using different strategies
        """
        predictions = []
        
        for i in range(num_predictions):
            np.random.seed(int(time.time()) + i)  # Different seed for each prediction
            prediction = self.predict_next_draw()
            if prediction and len(prediction) == 6:
                predictions.append(prediction)
                
        return predictions
    
    def _save_incremental_data(self, new_contests: List[Dict], csv_file: str):
        """
        Save new contest data incrementally to CSV file
        """
        if not new_contests:
            return
            
        try:
            # Process new contests into DataFrame format
            processed_data = []
            for contest in new_contests:
                numbers = contest.get('dezenas', [])
                if len(numbers) == 6:
                    processed_data.append({
                        'concurso': contest.get('concurso'),
                        'data': contest.get('data'),
                        'numero_1': int(numbers[0]),
                        'numero_2': int(numbers[1]),
                        'numero_3': int(numbers[2]),
                        'numero_4': int(numbers[3]),
                        'numero_5': int(numbers[4]),
                        'numero_6': int(numbers[5])
                    })
            
            if processed_data:
                new_df = pd.DataFrame(processed_data)
                
                # Append to existing CSV or create new one
                if os.path.exists(csv_file):
                    new_df.to_csv(csv_file, mode='a', header=False, index=False)
                else:
                    new_df.to_csv(csv_file, index=False)
                    
        except Exception as e:
            print(f"⚠️  Erro ao salvar dados incrementalmente: {e}")
    
    def save_data(self, filename: str = "mega_sena_data.csv"):
        """
        Save collected data to CSV file
        """
        if self.df is None or self.df.empty:
            print("❌ Nenhum dado para salvar!")
            return
            
        try:
            self.df.to_csv(filename, index=False)
            print(f"💾 Dados salvos em: {filename}")
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
    
    def create_visualizations(self):
        """
        Create data visualizations
        """
        if self.df is None or self.df.empty:
            print("❌ Nenhum dado para visualizar!")
            return
            
        print("📊 Criando visualizações...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise Estatística da Mega Sena', fontsize=16, fontweight='bold')
        
        # 1. Number frequency
        freq_analysis = self.analyze_frequency()
        frequencies = freq_analysis['frequencias']
        
        numbers = list(range(1, 61))
        freqs = [frequencies.get(n, 0) for n in numbers]
        
        axes[0, 0].bar(numbers, freqs, alpha=0.7)
        axes[0, 0].set_title('Frequência de Cada Número')
        axes[0, 0].set_xlabel('Número')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Most drawn numbers
        most_drawn = freq_analysis['mais_sorteados'][:10]
        nums, counts = zip(*most_drawn)
        
        axes[0, 1].barh(range(len(nums)), counts, alpha=0.7)
        axes[0, 1].set_yticks(range(len(nums)))
        axes[0, 1].set_yticklabels(nums)
        axes[0, 1].set_title('Top 10 Números Mais Sorteados')
        axes[0, 1].set_xlabel('Frequência')
        
        # 3. Even/Odd distribution
        pattern_analysis = self.analyze_patterns()
        even_odd_dist = pattern_analysis['distribuicao_pares_impares']
        
        labels = list(even_odd_dist.keys())
        values = list(even_odd_dist.values())
        
        axes[1, 0].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Distribuição Pares/Ímpares')
        
        # 4. Sum distribution
        sums = []
        for _, row in self.df.iterrows():
            numbers = [row[f'numero_{i}'] for i in range(1, 7)]
            sums.append(sum(numbers))
            
        axes[1, 1].hist(sums, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribuição da Soma dos Números')
        axes[1, 1].set_xlabel('Soma')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].axvline(np.mean(sums), color='red', linestyle='--', label=f'Média: {np.mean(sums):.1f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = "c:\\Users\\eduar\\Documentos\\projetos\\mega_sena\\mega_sena_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Gráficos salvos em: {plot_path}")
        
        plt.show()
    
    def print_analysis_report(self):
        """
        Print comprehensive analysis report
        """
        if self.df is None or self.df.empty:
            print("❌ Nenhum dado para analisar!")
            return
            
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE ANÁLISE DA MEGA SENA")
        print("="*60)
        
        # Basic statistics
        print(f"\n📈 ESTATÍSTICAS BÁSICAS:")
        print(f"   • Total de concursos analisados: {len(self.df)}")
        print(f"   • Período: {self.df['data'].min().strftime('%d/%m/%Y')} até {self.df['data'].max().strftime('%d/%m/%Y')}")
        print(f"   • Concursos acumulados: {self.df['acumulou'].sum()} ({(self.df['acumulou'].sum()/len(self.df)*100):.1f}%)")
        
        # Frequency analysis
        freq_analysis = self.analyze_frequency()
        print(f"\n🔢 ANÁLISE DE FREQUÊNCIA:")
        print(f"   • Média de aparições por número: {freq_analysis['media_aparicoes']:.1f}")
        
        print(f"\n   🔥 Top 10 números mais sorteados:")
        for i, (num, freq) in enumerate(freq_analysis['mais_sorteados'][:10], 1):
            print(f"      {i:2d}. Número {num:2d}: {freq:3d} vezes ({freq/len(self.df)*100:.1f}%)")
        
        print(f"\n   ❄️  Top 10 números menos sorteados:")
        for i, (num, freq) in enumerate(freq_analysis['menos_sorteados'][:10], 1):
            print(f"      {i:2d}. Número {num:2d}: {freq:3d} vezes ({freq/len(self.df)*100:.1f}%)")
        
        # Pattern analysis
        pattern_analysis = self.analyze_patterns()
        print(f"\n🎯 ANÁLISE DE PADRÕES:")
        print(f"   • Soma média dos números: {pattern_analysis['soma_media']:.1f}")
        print(f"   • Amplitude média: {pattern_analysis['amplitude_media']:.1f}")
        print(f"   • Média de números consecutivos: {pattern_analysis['consecutivos_media']:.2f}")
        
        print(f"\n   📊 Distribuição Pares/Ímpares:")
        for pattern, count in pattern_analysis['distribuicao_pares_impares'].most_common():
            percentage = (count / len(self.df)) * 100
            print(f"      {pattern}: {count:3d} vezes ({percentage:.1f}%)")
        
        # Predictions
        print(f"\n🔮 PREVISÕES PARA O PRÓXIMO SORTEIO:")
        predictions = self.generate_multiple_predictions(5)
        
        for i, prediction in enumerate(predictions, 1):
            numbers_str = " - ".join([f"{n:02d}" for n in prediction])
            print(f"   Jogo {i}: {numbers_str}")
        
        print("\n" + "="*60)
        print("⚠️  AVISO: Este programa é apenas para fins educacionais e de")
        print("   entretenimento. Loterias são jogos de azar e não há garantia")
        print("   de que as previsões estatísticas resultem em vitórias.")
        print("="*60)

def main():
    """
    Main function to run the Mega Sena analyzer
    """
    print("🎲 ANALISADOR E PREVISOR DA MEGA SENA")
    print("="*50)
    
    analyzer = MegaSenaAnalyzer()
    
    # Collect data
    if analyzer.collect_all_data():
        # Process data
        analyzer.process_data()
        
        if analyzer.df is not None and not analyzer.df.empty:
            # Save data
            analyzer.save_data()
            
            # Perform analysis
            analyzer.print_analysis_report()
            
            # Create visualizations
            try:
                analyzer.create_visualizations()
            except Exception as e:
                print(f"⚠️  Erro ao criar visualizações: {e}")
                print("   (Matplotlib pode não estar configurado corretamente)")
        else:
            print("❌ Falha no processamento dos dados!")
    else:
        print("❌ Falha na coleta de dados!")
        print("💡 Verifique sua conexão com a internet e tente novamente.")

if __name__ == "__main__":
    main()
