#!/usr/bin/env python3
"""
Mega Sena Web Dashboard
======================

Interface web moderna para visualizar análises da Mega Sena com atualização em tempo real.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import os
import threading
import time
import atexit
import signal
import sys
from datetime import datetime, timedelta
from collections import Counter
import plotly.graph_objs as go
import plotly.utils
from mega_sena_analyzer import MegaSenaAnalyzer

app = Flask(__name__)
app.secret_key = 'mega_sena_dashboard_2024'

# Application lock system
LOCK_FILE = "app.lock"
PID_FILE = "app.pid"

# ...restante do código...

# Adiciona a função e o endpoint após a criação do app
def generate_interval_predictions(df, num_predictions=8):
    """Gera previsões dos próximos jogos usando média dos intervalos de aparição de cada número"""
    if df.empty:
        return []

    # Para cada número, calcula os índices dos concursos em que apareceu
    number_intervals = {}
    number_last_index = {}
    for num in range(1, 61):
        appearances = []
        for idx, row in df.iterrows():
            if num in [row[f'numero_{i}'] for i in range(1, 7)]:
                appearances.append(idx)
        if len(appearances) > 1:
            intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
            avg_interval = np.mean(intervals)
            number_intervals[num] = avg_interval
            number_last_index[num] = appearances[-1]
        elif len(appearances) == 1:
            # Se só apareceu uma vez, assume intervalo médio igual à média global
            number_intervals[num] = None
            number_last_index[num] = appearances[-1]
        else:
            number_intervals[num] = None
            number_last_index[num] = None

    # Média global para números com poucos dados
    global_avg = np.mean([v for v in number_intervals.values() if v is not None])

    # Prever próximos 8 jogos
    last_contest = len(df) - 1
    predictions = []
    for i in range(1, num_predictions+1):
        contest_index = last_contest + i
        # Para cada número, calcula quando ele deve aparecer de novo
        predicted_appearance = {}
        for num in range(1, 61):
            if number_last_index[num] is not None:
                interval = number_intervals[num] if number_intervals[num] is not None else global_avg
                predicted_appearance[num] = number_last_index[num] + interval
            else:
                predicted_appearance[num] = contest_index + global_avg
        # Seleciona os 6 números cuja previsão de aparição está mais próxima do concurso atual
        sorted_nums = sorted(predicted_appearance.items(), key=lambda x: abs(x[1] - contest_index))
        selected = [num for num, _ in sorted_nums[:6]]
        predictions.append({
            'numbers': sorted(selected),
            'contest_estimate': contest_index + 1
        })
    return predictions

@app.route('/api/predictions-interval')
def generate_interval_predictions(df, num_predictions=8):
    """Gera previsões dos próximos jogos usando média dos intervalos de aparição de cada número"""
    if df.empty:
        return []

    # Para cada número, calcula os índices dos concursos em que apareceu
    number_intervals = {}
    number_last_index = {}
    for num in range(1, 61):
        appearances = []
        for idx, row in df.iterrows():
            if num in [row[f'numero_{i}'] for i in range(1, 7)]:
                appearances.append(idx)
        if len(appearances) > 1:
            intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
            avg_interval = np.mean(intervals)
            number_intervals[num] = avg_interval
            number_last_index[num] = appearances[-1]
        elif len(appearances) == 1:
            # Se só apareceu uma vez, assume intervalo médio igual à média global
            number_intervals[num] = None
            number_last_index[num] = appearances[-1]
        else:
            number_intervals[num] = None
            number_last_index[num] = None

    # Média global para números com poucos dados
    global_avg = np.mean([v for v in number_intervals.values() if v is not None])

    # Prever próximos 8 jogos
    last_contest = len(df) - 1
    predictions = []
    for i in range(1, num_predictions+1):
        contest_index = last_contest + i
        # Para cada número, calcula quando ele deve aparecer de novo
        predicted_appearance = {}
        for num in range(1, 61):
            if number_last_index[num] is not None:
                interval = number_intervals[num] if number_intervals[num] is not None else global_avg
                predicted_appearance[num] = number_last_index[num] + interval
            else:
                predicted_appearance[num] = contest_index + global_avg
        # Seleciona os 6 números cuja previsão de aparição está mais próxima do concurso atual
        sorted_nums = sorted(predicted_appearance.items(), key=lambda x: abs(x[1] - contest_index))
        selected = [num for num, _ in sorted_nums[:6]]
        predictions.append({
            'numbers': sorted(selected),
            'contest_estimate': contest_index + 1
        })
    return predictions

@app.route('/api/predictions-interval')
def get_interval_predictions():
    """Endpoint para previsões dos próximos 8 jogos usando média dos intervalos"""
    df = load_data()
    predictions = generate_interval_predictions(df, num_predictions=8)
    return jsonify({'predictions_interval': predictions})
#!/usr/bin/env python3
"""
Mega Sena Web Dashboard
======================

Interface web moderna para visualizar análises da Mega Sena com atualização em tempo real.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import os
import threading
import time
import atexit
import signal
import sys
from datetime import datetime, timedelta
from collections import Counter
import plotly.graph_objs as go
import plotly.utils
from mega_sena_analyzer import MegaSenaAnalyzer

app = Flask(__name__)
app.secret_key = 'mega_sena_dashboard_2024'

# Application lock system
LOCK_FILE = "app.lock"
PID_FILE = "app.pid"

def create_lock():
    """Create application lock to prevent multiple instances"""
    try:
        # Check if lock file exists and if the process is still running
        if os.path.exists(LOCK_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if process is still running (Windows)
                import subprocess
                try:
                    result = subprocess.run(['tasklist', '/FI', f'PID eq {old_pid}'], 
                                          capture_output=True, text=True)
                    if str(old_pid) in result.stdout:
                        print(f"❌ Aplicação já está rodando (PID: {old_pid})")
                        print("   Para parar a instância anterior, execute:")
                        print(f"   taskkill /F /PID {old_pid}")
                        sys.exit(1)
                    else:
                        # Process not running, remove stale lock
                        os.remove(LOCK_FILE)
                        if os.path.exists(PID_FILE):
                            os.remove(PID_FILE)
                except:
                    # If we can't check, assume it's stale and remove
                    os.remove(LOCK_FILE)
                    if os.path.exists(PID_FILE):
                        os.remove(PID_FILE)
            except:
                # If we can't read PID file, remove lock
                os.remove(LOCK_FILE)
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
        
        # Create new lock
        current_pid = os.getpid()
        with open(LOCK_FILE, 'w') as f:
            f.write(f"Dashboard Mega Sena - PID: {current_pid}")
        
        with open(PID_FILE, 'w') as f:
            f.write(str(current_pid))
        
        print(f"🔒 Lock criado com sucesso (PID: {current_pid})")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar lock: {e}")
        return False

def remove_lock():
    """Remove application lock"""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        print("🔓 Lock removido com sucesso")
    except Exception as e:
        print(f"⚠️  Erro ao remover lock: {e}")

def signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\n🛑 Recebido sinal {signum}, finalizando aplicação...")
    remove_lock()
    sys.exit(0)

# Register signal handlers and cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(remove_lock)

# Global variables with file persistence
import pickle
import threading

status_file = "update_status.pkl"
status_lock = threading.Lock()

def load_update_status():
    """Load update status from file"""
    try:
        if os.path.exists(status_file):
            with open(status_file, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return {
        'is_updating': False,
        'progress': 0,
        'total': 0,
        'current_contest': 0,
        'message': 'Pronto para atualizar'
    }

def save_update_status(status):
    """Save update status to file"""
    try:
        # Ensure all values are serializable
        safe_status = {
            'is_updating': bool(status.get('is_updating', False)),
            'progress': int(status.get('progress', 0)) if status.get('progress') is not None else 0,
            'total': int(status.get('total', 0)) if status.get('total') is not None else 0,
            'current_contest': int(status.get('current_contest', 0)) if status.get('current_contest') is not None else 0,
            'message': str(status.get('message', 'Pronto para atualizar'))
        }
        
        with open(status_file, 'wb') as f:
            pickle.dump(safe_status, f)
    except Exception as e:
        print(f"Erro ao salvar status: {e}")

# Load initial status
update_status = load_update_status()

def load_data():
    """Load data from CSV file"""
    csv_file = "mega_sena_data.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # Aceita datas em formato misto (ISO8601 ou brasileiro)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
            return df
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def analyze_frequency(df):
    """Analyze number frequency"""
    if df.empty:
        return {}
    
    all_numbers = []
    for i in range(1, 7):
        all_numbers.extend(df[f'numero_{i}'].tolist())
    
    frequency = Counter(all_numbers)
    
    return {
        'most_frequent': dict(frequency.most_common(10)),
        'least_frequent': dict(frequency.most_common()[-10:]),
        'all_frequencies': dict(frequency)
    }

def create_frequency_chart(df):
    """Create frequency chart"""
    if df.empty:
        return {}
    
    freq_data = analyze_frequency(df)
    all_freq = freq_data['all_frequencies']
    
    numbers = list(range(1, 61))
    frequencies = [all_freq.get(num, 0) for num in numbers]
    
    fig = go.Figure(data=[
        go.Bar(
            x=numbers,
            y=frequencies,
            marker_color='rgba(55, 128, 191, 0.7)',
            marker_line_color='rgba(55, 128, 191, 1.0)',
            marker_line_width=1
        )
    ])
    
    fig.update_layout(
        title='Frequência dos Números da Mega Sena',
        xaxis_title='Números',
        yaxis_title='Frequência',
        template='plotly_white',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_timeline_chart(df):
    """Create timeline chart showing contests over time"""
    if df.empty:
        return {}
    
    # Group by year
    df_timeline = df.copy()
    df_timeline['year'] = df_timeline['data'].dt.year
    contests_per_year = df_timeline.groupby('year').size()
    
    fig = go.Figure(data=[
        go.Scatter(
            x=contests_per_year.index,
            y=contests_per_year.values,
            mode='lines+markers',
            marker_color='rgba(255, 127, 14, 0.8)',
            line_color='rgba(255, 127, 14, 1.0)'
        )
    ])
    
    fig.update_layout(
        title='Concursos da Mega Sena por Ano',
        xaxis_title='Ano',
        yaxis_title='Número de Concursos',
        template='plotly_white',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def update_data_background():
    """Update data in background with real progress tracking"""
    global update_status
    
    try:
        with status_lock:
            update_status['is_updating'] = True
            update_status['message'] = 'Iniciando atualização...'
            update_status['progress'] = 0
            update_status['total'] = 0
            update_status['current_contest'] = 0
            save_update_status(update_status)
        
        # Create custom analyzer with progress tracking
        analyzer = MegaSenaAnalyzer()
        
        # Override the collect method to track progress
        original_collect = analyzer.collect_all_data

        
        def collect_with_real_progress():
            """Collect data with real progress tracking"""
            import requests
            import pandas as pd
            import os
            import time
            
            try:
                # Check existing data
                csv_file = "mega_sena_data.csv"
                last_contest = 0
                
                if os.path.exists(csv_file):
                    try:
                        existing_df = pd.read_csv(csv_file)
                        if not existing_df.empty:
                            last_contest = existing_df['concurso'].max()
                            
                            # Load existing data
                            for _, row in existing_df.iterrows():
                                contest_data = {
                                    'concurso': row['concurso'],
                                    'data': row['data'],
                                    'dezenas': [str(row[f'numero_{i}']) for i in range(1, 7)]
                                }
                                analyzer.data.append(contest_data)
                    except Exception as e:
                        print(f"Erro ao carregar dados existentes: {e}")
                        last_contest = 0
                
                # Get latest contest from API
                latest_url = f"{analyzer.api_base_url}/megasena/latest"
                response = requests.get(latest_url, timeout=30)
                
                if response.status_code != 200:
                    return len(analyzer.data) > 0
                    
                latest_data = response.json()
                latest_contest = latest_data.get('concurso', 0)
                
                # Calculate what needs to be collected
                start_contest = last_contest + 1 if last_contest > 0 else 1
                
                if start_contest > latest_contest:
                    with status_lock:
                        update_status['message'] = 'Dados já estão atualizados!'
                        save_update_status(update_status)
                    return len(analyzer.data) > 0
                
                total_to_collect = latest_contest - start_contest + 1
                
                with status_lock:
                    update_status['total'] = total_to_collect
                    update_status['message'] = f'Coletando {total_to_collect} novos concursos...'
                    save_update_status(update_status)
                
                # Collect new data with progress tracking
                collected = 0
                errors = 0
                new_contests = []
                
                for contest_num in range(start_contest, latest_contest + 1):
                    try:
                        contest_url = f"{analyzer.api_base_url}/megasena/{contest_num}"
                        response = requests.get(contest_url, timeout=10)
                        
                        if response.status_code == 200:
                            contest_data = response.json()
                            analyzer.data.append(contest_data)
                            new_contests.append(contest_data)
                            collected += 1
                            
                            # Update progress
                            progress_percent = (collected / total_to_collect) * 100
                            
                            with status_lock:
                                update_status['progress'] = collected
                                update_status['current_contest'] = contest_num
                                update_status['message'] = f'Coletando concurso {contest_num}... ({collected}/{total_to_collect})'
                                save_update_status(update_status)
                            
                            # Save incrementally every 5 contests
                            if collected % 5 == 0 or contest_num == latest_contest:
                                analyzer._save_incremental_data(new_contests, csv_file)
                                new_contests = []
                        else:
                            errors += 1
                            
                        time.sleep(0.1)
                        
                    except Exception as e:
                        errors += 1
                        if errors > 20:
                            with status_lock:
                                update_status['message'] = 'Muitos erros na coleta. Parando...'
                                save_update_status(update_status)
                            if new_contests:
                                analyzer._save_incremental_data(new_contests, csv_file)
                            break
                
                # Save any remaining contests
                if new_contests:
                    analyzer._save_incremental_data(new_contests, csv_file)
                
                return len(analyzer.data) > 0
                
                # Save any remaining contests
                if new_contests:
                    analyzer._save_incremental_data(new_contests, csv_file)
                
                return len(analyzer.data) > 0
                
            except Exception as e:
                with status_lock:
                    update_status['message'] = f'Erro na coleta: {str(e)}'
                    save_update_status(update_status)
                return len(analyzer.data) > 0
        
        # Use our custom collection method
        if collect_with_real_progress():
            with status_lock:
                update_status['message'] = 'Processando dados...'
                save_update_status(update_status)
            
            analyzer.process_data()
            
            with status_lock:
                update_status['message'] = 'Atualização concluída!'
                save_update_status(update_status)
        else:
            with status_lock:
                update_status['message'] = 'Erro na atualização'
                save_update_status(update_status)
            
    except Exception as e:
        with status_lock:
            update_status['message'] = f'Erro: {str(e)}'
            save_update_status(update_status)
    finally:
        with status_lock:
            update_status['is_updating'] = False
            update_status['progress'] = 0
            update_status['total'] = 0
            save_update_status(update_status)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """Get current data and statistics"""
    df = load_data()
    
    if df.empty:
        return jsonify({
            'total_contests': 0,
            'latest_contest': 0,
            'latest_date': '',
            'frequency_chart': {},
            'timeline_chart': {},
            'top_numbers': [],
            'predictions': []
        })
    
    freq_data = analyze_frequency(df)
    
    return jsonify({
        'total_contests': int(len(df)),
        'latest_contest': int(df['concurso'].max()),
        'latest_date': df['data'].max().strftime('%d/%m/%Y'),
        'frequency_chart': create_frequency_chart(df),
        'timeline_chart': create_timeline_chart(df),
        'top_numbers': [int(x) for x in list(freq_data['most_frequent'].keys())[:6]],
        'predictions': generate_predictions(df)
    })

@app.route('/api/update', methods=['POST'])
def start_update():
    """Start data update in background"""
    global update_status
    
    if update_status['is_updating']:
        return jsonify({'status': 'already_updating'})
    
    # Start update in background thread
    thread = threading.Thread(target=update_data_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/update-status')
def get_update_status():
    """Get current update status"""
    try:
        with status_lock:
            # Reload status from file to get latest state
            current_status = load_update_status()
            update_status.update(current_status)
            
            # Ensure all values are JSON serializable
            safe_status = {
                'is_updating': bool(update_status.get('is_updating', False)),
                'progress': int(update_status.get('progress', 0)),
                'total': int(update_status.get('total', 0)),
                'current_contest': int(update_status.get('current_contest', 0)),
                'message': str(update_status.get('message', 'Pronto para atualizar'))
            }
            
            return jsonify(safe_status)
    except Exception as e:
        print(f"Erro no endpoint update-status: {e}")
        # Return safe default status
        return jsonify({
            'is_updating': False,
            'progress': 0,
            'total': 0,
            'current_contest': 0,
            'message': f'Erro ao obter status: {str(e)}'
        })

def get_next_draw_dates(num_draws=5):
    """Calculate the next draw dates (Wednesdays and Saturdays)"""
    try:
        now = datetime.now()
        next_dates = []
        current_date = now.date()
        
        # Find next draw date
        days_ahead = 0
        while len(next_dates) < num_draws:
            check_date = current_date + timedelta(days=days_ahead)
            weekday = check_date.weekday()
            
            # Wednesday = 2, Saturday = 5
            if weekday == 2 or weekday == 5:  # Wednesday or Saturday
                # Only add if it's today and after 8 PM, or future dates
                if check_date > current_date or (check_date == current_date and now.hour >= 20):
                    next_dates.append(check_date)
            
            days_ahead += 1
            
            # Safety break to avoid infinite loop
            if days_ahead > 30:
                break
        
        return next_dates
    except Exception as e:
        print(f"Erro ao calcular próximas datas: {e}")
        return []

def generate_interval_predictions(df, num_predictions=8):
    """Generate predictions for next draws with dates and interval-based analysis"""
    if df.empty:
        return []
    
    try:
        freq_data = analyze_frequency(df)
        all_freq = freq_data['all_frequencies']
        
        # Get numbers with their intervals
        number_intervals = {}
        for num in range(1, 61):
            appearances = []
            for idx, row in df.iterrows():
                if num in [row[f'numero_{i}'] for i in range(1, 7)]:
                    appearances.append(idx)
            if len(appearances) > 1:
                intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                avg_interval = np.mean(intervals)
                number_intervals[num] = avg_interval
            else:
                number_intervals[num] = None
        
        # Get next draw dates
        next_dates = get_next_draw_dates(num_predictions)
        
        predictions = []
        for i in range(num_predictions):
            prediction = []
            for num, interval in number_intervals.items():
                if interval is not None:
                    # Calculate next predicted appearance
                    next_appearance = df.index[-1] + interval
                    # If next appearance is before the current date, skip
                    if next_appearance < next_dates[i].toordinal():
                        continue
                    # Add the number to the prediction
                    prediction.append(num)
            # Ensure we have exactly 6 unique numbers
            prediction = list(set(prediction))[:6]
            while len(prediction) < 6:
                # Fill with random numbers from 1-60 not already in prediction
                available = [n for n in range(1, 61) if n not in prediction]
                if available:
                    prediction.append(np.random.choice(available))
            
            prediction_data = {
                'numbers': sorted([int(x) for x in prediction[:6]]),
                'date': next_dates[i].strftime('%d/%m/%Y') if i < len(next_dates) else 'Data não disponível',
                'weekday': next_dates[i].strftime('%A') if i < len(next_dates) else '',
                'contest_estimate': None  # Will be calculated based on current data
            }
            
            predictions.append(prediction_data)
        
        return predictions
    except Exception as e:
        print(f"Erro ao gerar previsões com análise de intervalos: {e}")
        return []

if __name__ == '__main__':
    print("🌐 Iniciando Dashboard da Mega Sena...")
    
    # Create application lock to prevent multiple instances
    if not create_lock():
        print("❌ Falha ao criar lock da aplicação")
        sys.exit(1)
    
    try:
        print("📊 Acesse: http://localhost:5000")
        print("🔒 Aplicação protegida contra múltiplas execuções")
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Aplicação interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro na aplicação: {e}")
    finally:
        remove_lock()
