import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

# Set page configuration
st.set_page_config(
    page_title="Análise Comparativa de Óbitos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "https://apisidra.ibge.gov.br/"
TABELA = 2683  # Óbitos por estado civil, natureza do óbito, etc.
VARIAVEL = '343'  # Número de óbitos ocorridos no ano
NATUREZA_OBITO = '99818'  # Não natural
ESTADO_CIVIL_CASADO = '99197'  # Casado(a)
ESTADO_CIVIL_NAO_CASADOS = ['78090', '78092', '78093', '78094', '99195', '99217']  # Códigos para não casados
NIVEL_TERRITORIAL = 'N7'  # Região Metropolitana
PERIODOS = [str(year) for year in range(2003, 2023)]  # 2003 a 2022

# Defined list of metropolitan regions
REGIOES_METROPOLITANAS = {
    '2701': 'Maceió',
    '1301': 'Manaus',
    '1601': 'Macapá',
    '2901': 'Salvador',
    '2301': 'Fortaleza',
    '3201': 'Grande Vitória',
    '5201': 'Goiânia',
    '2101': 'Grande São Luís',
    '3101': 'Belo Horizonte',
    '5101': 'Vale do Rio Cuiabá',
    '1501': 'Belém',
    '2501': 'João Pessoa',
    '2601': 'Recife',
    '4101': 'Curitiba',
    '3301': 'Rio de Janeiro',
    '2401': 'Natal',
    '4301': 'Porto Alegre',
    '4201': 'Florianópolis',
    '2801': 'Aracaju',
    '3501': 'São Paulo'
}

# Execute SIDRA API query for unnatural deaths among married people
def consultar_obitos_casados(rm_id, rm_nome):
    """
    Consulta a API SIDRA para obter dados da variável 343 (número de óbitos)
    para óbitos não naturais de pessoas casadas em uma região metropolitana.
    """
    url = f"{API_URL}values/t/{TABELA}/v/{VARIAVEL}/p/{','.join(PERIODOS)}/c9832/{ESTADO_CIVIL_CASADO}/c1836/{NATUREZA_OBITO}/{NIVEL_TERRITORIAL}/{rm_id}/f/n"

    with st.spinner(f"Consultando API de óbitos não naturais para casados em {rm_nome}..."):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            st.warning(f"Erro na consulta à API para {rm_nome}: {str(e)}")

            # Tentativa alternativa - usando N1 (Brasil)
            st.info("Tentando consulta alternativa usando dados do Brasil...")
            try:
                url_alt = f"{API_URL}values/t/{TABELA}/v/{VARIAVEL}/p/{','.join(PERIODOS)}/c9832/{ESTADO_CIVIL_CASADO}/c1836/{NATUREZA_OBITO}/n1/1/f/n"
                resp_alt = requests.get(url_alt)
                resp_alt.raise_for_status()
                st.success("Consulta alternativa bem-sucedida (dados do Brasil)")
                return resp_alt.json()
            except:
                st.error("Consulta alternativa também falhou")
                return []

# Execute SIDRA API query for unnatural deaths among non-married people
def consultar_obitos_nao_casados(rm_id, rm_nome, estado_civil_codigo):
    """
    Consulta a API SIDRA para obter dados da variável 343 (número de óbitos)
    para óbitos não naturais de um grupo específico de não casados em uma região metropolitana.
    """
    url = f"{API_URL}values/t/{TABELA}/v/{VARIAVEL}/p/{','.join(PERIODOS)}/c9832/{estado_civil_codigo}/c1836/{NATUREZA_OBITO}/{NIVEL_TERRITORIAL}/{rm_id}/f/n"

    with st.spinner(f"Consultando API para estado civil código {estado_civil_codigo} em {rm_nome}..."):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            st.warning(f"Erro na consulta à API para {rm_nome}: {str(e)}")
            return []

# Process API response into a dataframe
def processar_dados(data, rm_nome, tipo="casados"):
    """
    Processa a resposta da API SIDRA e extrai anos e valores.
    """
    if not data:
        st.info(f"Sem dados de óbitos não naturais ({tipo}) para {rm_nome}")
        return pd.DataFrame()

    # Extrair pares de ano e valor
    anos_valores = []

    for item in data:
        try:
            # Encontrar o campo que contém o ano
            ano = None
            for key, value in item.items():
                # Os períodos estão nos anos do estudo (2003 a 2022)
                if value in PERIODOS:
                    ano = value
                    break

            if not ano:
                continue

            # Obter o valor
            valor_str = item.get('V', '0')  # Valor

            # Tratar caracteres especiais
            if valor_str in ['-', 'X', '..', '...']:
                continue

            # Converter para float
            if isinstance(valor_str, str):
                valor_str = valor_str.replace(',', '.')
            valor = float(valor_str)

            # Obter unidade de medida
            unidade = item.get('MN', 'Pessoas')

            anos_valores.append({'Ano': ano, 'Valor': valor, 'Unidade': unidade})

        except (ValueError, TypeError) as e:
            st.warning(f"Erro ao processar item: {e}")
            continue

    # Criar DataFrame
    df = pd.DataFrame(anos_valores)

    # Verificar se temos dados
    if df.empty:
        st.warning(f"Nenhum dado válido encontrado para {rm_nome} ({tipo})")
        return df

    st.success(f"Dados processados para {rm_nome} ({tipo}): {len(df)} registros")
    return df

# Create comparative time series chart for unnatural deaths
def criar_grafico_comparativo(df_casados, df_nao_casados, rm_nome, rm_codigo):
    """
    Cria um gráfico comparativo mostrando a evolução dos óbitos não naturais
    entre pessoas casadas e não casadas ao longo dos anos.
    """
    if df_casados.empty and df_nao_casados.empty:
        st.warning(f"Sem dados para criar gráfico comparativo para {rm_nome}")
        return

    # Criar figura limpa com fundo branco
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')

    # Definir anos a serem plotados
    all_years = set()
    if not df_casados.empty:
        all_years.update(df_casados['Ano'])
    if not df_nao_casados.empty:
        all_years.update(df_nao_casados['Ano'])

    all_years = sorted(list(all_years))

    # Definir valores mínimo e máximo para o eixo y
    min_values = []
    max_values = []

    # Plotar dados para casados
    if not df_casados.empty:
        df_sorted = df_casados.sort_values('Ano')
        line1 = ax.plot(df_sorted['Ano'], df_sorted['Valor'], '-',
                      color='darkred',
                      linewidth=2.5,
                      marker='o',
                      markersize=8,
                      label='Casados')[0]

        # Adicionar rótulos
        for x, y in zip(df_sorted['Ano'], df_sorted['Valor']):
            if int(x) % 3 == 0:  # Adicionar rótulos apenas a cada 3 anos
                ax.annotate(f'{int(y)}',
                          xy=(x, y),
                          xytext=(0, 10),
                          textcoords='offset points',
                          ha='center',
                          fontsize=9,
                          fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.7))

        min_values.append(df_sorted['Valor'].min())
        max_values.append(df_sorted['Valor'].max())

    # Plotar dados para não casados
    if not df_nao_casados.empty:
        df_sorted = df_nao_casados.sort_values('Ano')
        line2 = ax.plot(df_sorted['Ano'], df_sorted['Valor'], '-',
                      color='navy',
                      linewidth=2.5,
                      marker='s',
                      markersize=8,
                      label='Não Casados')[0]

        # Adicionar rótulos
        for x, y in zip(df_sorted['Ano'], df_sorted['Valor']):
            if int(x) % 3 == 0:  # Adicionar rótulos apenas a cada 3 anos
                ax.annotate(f'{int(y)}',
                          xy=(x, y),
                          xytext=(0, -25),
                          textcoords='offset points',
                          ha='center',
                          fontsize=9,
                          fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="navy", alpha=0.7))

        min_values.append(df_sorted['Valor'].min())
        max_values.append(df_sorted['Valor'].max())

    # Configurar título e rótulos dos eixos
    plt.title(f'Óbitos Não Naturais: Casados vs. Não Casados\n{rm_nome} (Código {rm_codigo})\n2003-2022',
             fontsize=14, fontweight='bold')
    plt.xlabel('Ano', fontsize=12, fontweight='bold')
    plt.ylabel('Número de Óbitos', fontsize=12, fontweight='bold')

    # Configurar ticks do eixo x para mostrar anos selecionados
    # Mostra anos a cada 3 anos para evitar sobrecarga
    anos_mostrar = [str(year) for year in range(2003, 2023, 3)]
    plt.xticks(anos_mostrar, anos_mostrar, rotation=45, fontsize=10)

    # Calcular intervalo adequado para as linhas de grade horizontais
    if min_values and max_values:
        y_min = max(0, min(min_values) * 0.9)
        y_max = max(max_values) * 1.1

        # Calcular ticks adequados para as linhas horizontais
        range_size = y_max - y_min
        if range_size <= 10:
            step = 1
        elif range_size <= 50:
            step = 5
        elif range_size <= 100:
            step = 10
        elif range_size <= 500:
            step = 50
        else:
            step = 100

        # Criar ticks começando de um número redondo
        start = np.floor(y_min / step) * step
        ticks = np.arange(start, y_max + step, step)

        plt.ylim(y_min, y_max)
        plt.yticks(ticks, fontsize=10)

    # Adicionar linhas de grade horizontais proeminentes
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    # Remover bordas superior e direita para um visual mais limpo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adicionar tendência para ambas as séries se houver dados suficientes
    for df, cor, nome in [(df_casados, 'darkred', 'Casados'), (df_nao_casados, 'navy', 'Não Casados')]:
        if not df.empty and len(df) > 1:
            # Converter anos para valores numéricos para cálculo de tendência
            df['Ano_Num'] = df['Ano'].astype(int)
            x = df['Ano_Num'].values
            y = df['Valor'].values

            # Calcular linha de tendência
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            # Plotar linha de tendência
            anos_num = np.array([int(ano) for ano in df['Ano']])
            ax.plot(df['Ano'], p(anos_num), '--', color=cor, linewidth=1.5,
                   label=f'Tendência {nome}: {z[0]:.1f}/ano')

    # Adicionar legenda
    plt.legend(loc='best', fontsize=10)

    # Adicionar texto com a razão entre não casados e casados para o último ano
    if not df_casados.empty and not df_nao_casados.empty:
        ultimo_ano_casados = df_casados.sort_values('Ano').iloc[-1]
        ultimo_ano_nao_casados = df_nao_casados.sort_values('Ano').iloc[-1]

        if ultimo_ano_casados['Ano'] == ultimo_ano_nao_casados['Ano'] and ultimo_ano_casados['Valor'] > 0:
            razao = ultimo_ano_nao_casados['Valor'] / ultimo_ano_casados['Valor']
            texto_razao = f"Razão Não Casados/Casados em {ultimo_ano_casados['Ano']}: {razao:.1f}x"
            plt.figtext(0.15, 0.02, texto_razao, ha='left', fontsize=11,
                       color='black', weight='bold',
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    plt.tight_layout()
    
    return fig

# Create a summary chart showing ratios between non-married and married
def criar_grafico_resumo_razao(all_data):
    """
    Cria um gráfico resumo mostrando a razão entre óbitos não naturais
    de não casados e casados para cada região metropolitana.
    """
    if not all_data:
        st.warning("Sem dados suficientes para criar gráfico resumo")
        return None

    # Calcular razão para cada região no último ano disponível
    razoes = []

    for rm_nome, dados in all_data.items():
        if 'casados' not in dados or 'nao_casados' not in dados:
            continue

        df_casados = dados['casados']
        df_nao_casados = dados['nao_casados']

        if df_casados.empty or df_nao_casados.empty:
            continue

        # Obter último ano comum
        ultimo_ano_casados = df_casados.sort_values('Ano').iloc[-1]['Ano']
        ultimo_ano_nao_casados = df_nao_casados.sort_values('Ano').iloc[-1]['Ano']

        if ultimo_ano_casados != ultimo_ano_nao_casados:
            continue

        ultimo_ano = ultimo_ano_casados

        # Obter valores para o último ano
        valor_casados = df_casados[df_casados['Ano'] == ultimo_ano]['Valor'].iloc[0]
        valor_nao_casados = df_nao_casados[df_nao_casados['Ano'] == ultimo_ano]['Valor'].iloc[0]

        if valor_casados > 0:
            razao = valor_nao_casados / valor_casados
            razoes.append({
                'RM': rm_nome,
                'Razão Não Casados/Casados': razao,
                'Valor Casados': valor_casados,
                'Valor Não Casados': valor_nao_casados,
                'Ano': ultimo_ano
            })

    if not razoes:
        st.warning("Não há dados suficientes para calcular razões")
        return None

    # Criar DataFrame e ordenar
    df_razao = pd.DataFrame(razoes)
    df_razao = df_razao.sort_values('Razão Não Casados/Casados', ascending=False)

    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Plotar barras horizontais
    bars = ax.barh(df_razao['RM'], df_razao['Razão Não Casados/Casados'], color='purple')

    # Adicionar rótulos de valor
    for i, bar in enumerate(bars):
        valor = df_razao['Razão Não Casados/Casados'].iloc[i]
        ax.text(valor + 0.1, bar.get_y() + bar.get_height()/2, f"{valor:.1f}x",
               va='center', ha='left', fontsize=10, fontweight='bold')

    # Configurar título e rótulos
    plt.title(f'Razão entre Óbitos Não Naturais de Não Casados e Casados ({df_razao["Ano"].iloc[0]})',
             fontsize=14, fontweight='bold')
    plt.xlabel('Razão (Não Casados/Casados)', fontsize=12, fontweight='bold')
    plt.ylabel('Região Metropolitana', fontsize=12, fontweight='bold')

    # Adicionar grid horizontal
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Adicionar linha vertical em razão = 1
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1)

    # Remover bordas
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    return fig

# Streamlit UI
def main():
    st.title("Análise Comparativa de Óbitos Não Naturais: Casados vs. Não Casados")
    st.subheader("Por Região Metropolitana - 2003 a 2022")
    
    st.sidebar.title("Configurações")
    
    # Select regions to analyze
    selected_regions = st.sidebar.multiselect(
        "Selecione as Regiões Metropolitanas:",
        options=list(REGIOES_METROPOLITANAS.values()),
        default=["São Paulo", "Rio de Janeiro"]
    )
    
    # Create a button to start analysis
    if st.sidebar.button("Iniciar Análise"):
        if not selected_regions:
            st.warning("Por favor, selecione pelo menos uma região metropolitana.")
            return
        
        # Store data for all selected regions
        all_data = {}
        
        # Initialize a progress bar
        progress_bar = st.progress(0)
        total_steps = len(selected_regions) * (1 + len(ESTADO_CIVIL_NAO_CASADOS))
        current_step = 0
        
        # Process each selected metropolitan region
        for rm_nome in selected_regions:
            # Find the region code
            rm_id = None
            for id, nome in REGIOES_METROPOLITANAS.items():
                if nome == rm_nome:
                    rm_id = id
                    break
            
            if not rm_id:
                st.error(f"Código não encontrado para {rm_nome}")
                continue
            
            st.markdown(f"## {rm_nome} (Código {rm_id})")
            
            # Initialize data for this region
            all_data[rm_nome] = {}
            
            # Query data for married people
            data_casados = consultar_obitos_casados(rm_id, rm_nome)
            df_casados = processar_dados(data_casados, rm_nome, "casados")
            all_data[rm_nome]['casados'] = df_casados
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
            # Wait a bit to avoid overloading the API
            time.sleep(1)
            
            # For non-married people, get each category and sum them
            df_nao_casados_total = pd.DataFrame()
            
            for codigo in ESTADO_CIVIL_NAO_CASADOS:
                data_grupo = consultar_obitos_nao_casados(rm_id, rm_nome, codigo)
                df_grupo = processar_dados(data_grupo, rm_nome, f"estado civil {codigo}")
                
                if not df_grupo.empty:
                    if df_nao_casados_total.empty:
                        df_nao_casados_total = df_grupo.copy()
                    else:
                        # Merge data by year
                        df_merged = pd.merge(df_nao_casados_total, df_grupo, on=['Ano'], how='outer')
                        df_merged['Valor'] = df_merged['Valor_x'].fillna(0) + df_merged['Valor_y'].fillna(0)
                        df_merged['Unidade'] = df_merged['Unidade_x'].fillna(df_merged['Unidade_y'])
                        df_nao_casados_total = df_merged[['Ano', 'Valor', 'Unidade']].copy()
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Wait a bit to avoid overloading the API
                time.sleep(1)
            
            st.success(f"Dados agregados para não casados em {rm_nome}: {len(df_nao_casados_total)} registros")
            all_data[rm_nome]['nao_casados'] = df_nao_casados_total
            
            # Create comparative chart
            if not (df_casados.empty and df_nao_casados_total.empty):
                fig = criar_grafico_comparativo(df_casados, df_nao_casados_total, rm_nome, rm_id)
                st.pyplot(fig)
            else:
                st.warning(f"Sem dados suficientes para {rm_nome}.")
        
        # Create summary ratio chart
        if all_data:
            st.markdown("## Resumo de Razão entre Não Casados e Casados")
            fig_resumo = criar_grafico_resumo_razao(all_data)
            if fig_resumo:
                st.pyplot(fig_resumo)
        
        # Complete progress bar
        progress_bar.progress(1.0)
        st.success("Análise concluída!")
        
        # Show raw data in expander
        with st.expander("Ver dados brutos"):
            for rm_nome, dados in all_data.items():
                st.markdown(f"### {rm_nome}")
                
                st.markdown("#### Dados de Casados")
                if not dados['casados'].empty:
                    st.dataframe(dados['casados'])
                else:
                    st.info("Sem dados disponíveis")
                
                st.markdown("#### Dados de Não Casados")
                if not dados['nao_casados'].empty:
                    st.dataframe(dados['nao_casados'])
                else:
                    st.info("Sem dados disponíveis")

# Add a footer with information
def add_footer():
    st.markdown("---")
    st.markdown("""
    **Sobre esta aplicação:**
    
    Esta aplicação apresenta uma análise comparativa de óbitos não naturais entre pessoas casadas e não casadas 
    nas regiões metropolitanas do Brasil, com base nos dados do IBGE através da API SIDRA.
    
    Os dados são obtidos da tabela 2683, que contém informações sobre óbitos por estado civil e natureza do óbito.
    """)

if __name__ == "__main__":
    main()
    add_footer()
