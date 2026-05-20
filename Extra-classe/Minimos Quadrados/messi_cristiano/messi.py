import re
import time
import requests
import pandas as pd

from bs4 import BeautifulSoup


base_url = "https://www.ogol.com.br/jogador/lionel-messi/jogos"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "pt-BR,pt;q=0.9"
}

opcoes_epoca = [
    ("2026", "155"),
    ("2025", "154"),
    ("2024", "153"),
    ("2023", "152"),
    ("2021/22", "151"),
    ("2020/21", "150"),
    ("2019/20", "149"),
    ("2018/19", "148"),
    ("2017/18", "147"),
    ("2016/17", "146"),
    ("2015/16", "145"),
    ("2014/15", "144"),
    ("2013/14", "143"),
    ("2012/13", "142"),
    ("2011/12", "141"),
    ("2010/11", "140"),
    ("2009/10", "139"),
    ("2008/09", "138"),
    ("2007/08", "137"),
    ("2006/07", "136"),
    ("2005/06", "135"),
    ("2004/05", "134"),
]


def extrair_jogos_do_html(html, epoca):

    soup = BeautifulSoup(html, "html.parser")

    tabelas = soup.find_all("table")

    tabela_jogos = None

    for tabela in tabelas:

        texto = tabela.get_text(" ", strip=True)

        datas = re.findall(r"\d{4}/\d{2}/\d{2}", texto)

        if len(datas) > 5:
            tabela_jogos = tabela
            break

    if tabela_jogos is None:
        return []

    tbody = tabela_jogos.find("tbody")

    if tbody is None:
        return []

    dados = []

    for linha in tbody.find_all("tr"):

        colunas = linha.find_all("td")
        texto_linha = linha.get_text(" ", strip=True)

        match_data = re.search(r"\d{4}/\d{2}/\d{2}", texto_linha)

        if not match_data:
            continue

        textos_colunas = [c.get_text(" ", strip=True) for c in colunas]

        if len(textos_colunas) < 13:
            continue

        time_jogador = textos_colunas[4]

        if "barcelona b" in time_jogador.lower():
            continue

        if textos_colunas[8].strip().upper() == "NU":
            continue

        texto_gols = textos_colunas[12]
        coluna_gols_html = str(colunas[12]).lower()

        match_gols = re.search(r"x(\d+)", texto_gols)

        if match_gols:
            gols = int(match_gols.group(1))
        elif "zz-icn-fut-11" in coluna_gols_html:
            gols = 1
        else:
            gols = 0

        dados.append({
            "epoca": epoca,
            "data": match_data.group(0),
            "resultado": textos_colunas[0],
            "competicao": textos_colunas[2],
            "fase": textos_colunas[3],
            "time": time_jogador,
            "mando": textos_colunas[5],
            "adversario": textos_colunas[6],
            "placar": textos_colunas[7],
            "minutos": textos_colunas[8],
            "gols_no_jogo": gols
        })

    return dados


todos_os_dados = []

for epoca, valor in opcoes_epoca:

    url_epoca = f"{base_url}?epoca_id={valor}"

    resposta = requests.get(url_epoca, headers=headers)

    dados_epoca = extrair_jogos_do_html(resposta.text, epoca)

    todos_os_dados.extend(dados_epoca)

    time.sleep(1)


df = pd.DataFrame(todos_os_dados)

df["data"] = pd.to_datetime(df["data"], format="%Y/%m/%d")

df = df.drop_duplicates(
    subset=["data", "time", "adversario", "placar"]
)

df = df.sort_values("data")

df["jogo"] = range(1, len(df) + 1)

df["gols_acumulados"] = df["gols_no_jogo"].cumsum()

df.to_csv("messi_jogos_todos.csv", index=False, encoding="utf-8-sig")

print(f"CSV criado com {len(df)} jogos.")
print(f"Total de gols: {df['gols_no_jogo'].sum()}")