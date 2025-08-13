#!/usr/bin/env python3
"""
Script para limpar locks órfãos do Dashboard Mega Sena
=====================================================

Use este script se a aplicação não conseguir iniciar devido a locks órfãos.
"""

import os
import subprocess
import sys

LOCK_FILE = "app.lock"
PID_FILE = "app.pid"

def cleanup_locks():
    """Remove locks órfãos"""
    print("🧹 Limpando locks órfãos do Dashboard Mega Sena...")
    
    try:
        # Check if lock files exist
        if os.path.exists(LOCK_FILE) or os.path.exists(PID_FILE):
            
            # Try to read PID if exists
            old_pid = None
            if os.path.exists(PID_FILE):
                try:
                    with open(PID_FILE, 'r') as f:
                        old_pid = int(f.read().strip())
                    print(f"📋 PID encontrado no arquivo: {old_pid}")
                except:
                    print("⚠️  Não foi possível ler o arquivo PID")
            
            # Check if process is still running
            if old_pid:
                try:
                    result = subprocess.run(['tasklist', '/FI', f'PID eq {old_pid}'], 
                                          capture_output=True, text=True)
                    if str(old_pid) in result.stdout:
                        print(f"⚠️  Processo {old_pid} ainda está rodando!")
                        response = input("Deseja forçar o encerramento? (s/N): ")
                        if response.lower() in ['s', 'sim', 'y', 'yes']:
                            subprocess.run(['taskkill', '/F', '/PID', str(old_pid)])
                            print(f"🛑 Processo {old_pid} encerrado")
                        else:
                            print("❌ Operação cancelada")
                            return False
                    else:
                        print(f"✅ Processo {old_pid} não está mais rodando")
                except Exception as e:
                    print(f"⚠️  Erro ao verificar processo: {e}")
            
            # Remove lock files
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
                print("🗑️  Arquivo app.lock removido")
            
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
                print("🗑️  Arquivo app.pid removido")
            
            print("✅ Limpeza concluída! Agora você pode executar o Dashboard novamente.")
            
        else:
            print("✅ Nenhum lock órfão encontrado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante a limpeza: {e}")
        return False

def main():
    """Função principal"""
    print("🔧 LIMPADOR DE LOCKS - Dashboard Mega Sena")
    print("=" * 50)
    
    if cleanup_locks():
        print("\n🎉 Limpeza realizada com sucesso!")
        print("   Agora você pode executar: python app.py")
    else:
        print("\n❌ Falha na limpeza dos locks")
        sys.exit(1)

if __name__ == "__main__":
    main()
