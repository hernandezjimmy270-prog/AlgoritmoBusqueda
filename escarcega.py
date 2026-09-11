import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os

from scipy.spatial import cKDTree

from matplotlib.widgets import RadioButtons, Button

print("Inicio...")

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ox.settings.overpass_url = (
    "https://overpass.kumi.systems/api/interpreter"
)

RUTA_CACHE = "datos_escarcega.pkl"


# ============================================================
# 2. CARGAR DESDE LOCAL (RÁPIDO, SIN RED)
# ============================================================

if os.path.exists(RUTA_CACHE):

    print("Cargando datos desde local... (sin red)")

    with open(RUTA_CACHE, "rb") as archivo:

        datos = pickle.load(archivo)

    G = datos["G"]
    H = datos["H"]
    id_nuevo = datos["id_nuevo"]
    osm_por_id = datos["osm_por_id"]

    print("Datos cargados.")
    print("Nodos:", len(G.nodes))
    print("Conexiones:", len(G.edges))


# ============================================================
# 3. DESCARGAR (SOLO LA PRIMERA VEZ)
# ============================================================

if not os.path.exists(RUTA_CACHE):

    # BUSCAR ESCÁRCEGA

    print("Primera ejecución: descargando de internet (puede tardar)...")

    tags = {
        "boundary": "administrative",
        "name": "Escárcega"
    }

    gdf = ox.features_from_place(
        "Escárcega, Campeche, Mexico",
        tags
    )

    nivel_8 = gdf[
        gdf["admin_level"].astype(str) == "8"
    ]

    if nivel_8.empty:
        print("No se encontró el límite de Escárcega.")
        exit()

    poligono = nivel_8.geometry.iloc[0]

    print("Polígono encontrado.")


    # DESCARGAR CALLES

    print("Descargando calles (esto puede tardar)...")

    G = ox.graph_from_polygon(
        poligono,
        network_type="drive"
    )

    print("Mapa descargado.")
    print("Nodos:", len(G.nodes))
    print("Conexiones:", len(G.edges))


    # CREAR IDS DE NUESTRO PROYECTO

    nodos = list(G.nodes)

    id_nuevo = {}

    for i, nodo_osm in enumerate(nodos):
        id_nuevo[nodo_osm] = i


    osm_por_id = {
        id_proyecto: nodo_osm
        for nodo_osm, id_proyecto in id_nuevo.items()
    }


    print("\nIDs de nodos:")
    print("Desde:", 0)
    print("Hasta:", len(nodos) - 1)


    # CREAR GRAFO PARA LOS ALGORITMOS

    H = nx.DiGraph()

    for nodo_osm in G.nodes:

        nodo_id = id_nuevo[nodo_osm]

        H.add_node(nodo_id)


    for u, v, key, datos in G.edges(
        keys=True,
        data=True
    ):

        nodo_a = id_nuevo[u]
        nodo_b = id_nuevo[v]

        distancia = float(
            datos.get("length", 0)
        )


        if H.has_edge(nodo_a, nodo_b):

            actual = H[
                nodo_a
            ][
                nodo_b
            ]["weight"]


            if distancia < actual:

                H[
                    nodo_a
                ][
                    nodo_b
                ]["weight"] = distancia


        else:

            H.add_edge(
                nodo_a,
                nodo_b,
                weight=distancia
            )


    print("\nGrafo H creado.")
    print("Nodos en H:", len(H.nodes))
    print("Conexiones en H:", len(H.edges))


    # GUARDAR TODO EN LOCAL PARA LA PRÓXIMA VEZ

    print("\nGuardando datos en local...")

    with open(RUTA_CACHE, "wb") as archivo:

        pickle.dump(
            {
                "G": G,
                "H": H,
                "id_nuevo": id_nuevo,
                "osm_por_id": osm_por_id
            },
            archivo,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("Datos guardados. Las próximas veces no necesitará red.")


# ============================================================
# ÍNDICE DE NODOS PARA SELECCIÓN POR CLIC
# ============================================================

lista_nodos = list(G.nodes)

coordenadas = np.array(
    [
        [
            G.nodes[nodo_osm]["x"],
            G.nodes[nodo_osm]["y"]
        ]
        for nodo_osm in lista_nodos
    ]
)

arbol_nodos = cKDTree(
    coordenadas
)


# ============================================================
# MATRIZ DE PESOS
# ============================================================

print("Creando matriz de pesos (dispersa)...")

matriz_pesos = nx.to_scipy_sparse_array(
    H,
    nodelist=sorted(H.nodes),
    weight="weight"
)

matriz_pesos.setdiag(0)

print("\n--- MATRIZ DE PESOS (PRIMEROS 20 NODOS) ---")

matriz_mostrar = pd.DataFrame(
    matriz_pesos[:20, :20].toarray(),
    index=range(20),
    columns=range(20)
)

print(
    matriz_mostrar.to_string()
)


# ============================================================
# MAPA
# ============================================================

fig, ax = ox.plot_graph(
    G,
    figsize=(12, 12),
    node_size=15,
    node_color="red",
    edge_color="black",
    edge_linewidth=2,
    bgcolor="white",
    show=False,
    close=False
)

fig.subplots_adjust(right=0.75)

ax.set_title(
    "Haz clic sobre el mapa para elegir el nodo de INICIO, luego el FINAL."
)


# ============================================================
# FUNCIONES DE LOS ALGORITMOS
# ============================================================

def ejecutar_bfs(inicio, final):

    try:

        ruta = nx.shortest_path(
            H,
            inicio,
            final
        )

    except nx.NetworkXNoPath:

        return []


    return ruta


def ejecutar_dfs(inicio, final):

    visitados = set()

    padres = {}

    pila = [inicio]

    visitados.add(inicio)


    while pila:

        actual = pila.pop()


        if actual == final:

            break


        for vecino in reversed(
            list(H.neighbors(actual))
        ):

            if vecino not in visitados:

                visitados.add(vecino)

                padres[vecino] = actual

                pila.append(vecino)


    if final not in visitados:

        return []


    ruta = []

    actual = final


    while actual != inicio:

        ruta.append(actual)

        actual = padres[actual]


    ruta.append(inicio)

    ruta.reverse()


    return ruta


def ejecutar_dijkstra(inicio, final):

    try:

        ruta = nx.dijkstra_path(
            H,
            inicio,
            final,
            weight="weight"
        )

    except nx.NetworkXNoPath:

        return []


    return ruta


def distancia_entre_nodos(
    nodo_a,
    nodo_b
):

    osm_a = osm_por_id[nodo_a]

    osm_b = osm_por_id[nodo_b]


    datos = G.get_edge_data(
        osm_a,
        osm_b
    )


    if datos is None:

        return 0


    mejor_key = min(
        datos,
        key=lambda k:
        datos[k].get(
            "length",
            0
        )
    )


    return datos[
        mejor_key
    ].get(
        "length",
        0
    )


def ejecutar_a_estrella(
    inicio,
    final
):

    try:

        ruta = nx.astar_path(
            H,
            inicio,
            final,
            heuristic=(
                lambda a, b:
                distancia_geografica(
                    a,
                    b
                )
            ),
            weight="weight"
        )

    except nx.NetworkXNoPath:

        return []


    return ruta


def distancia_geografica(
    nodo_a,
    nodo_b
):

    osm_a = osm_por_id[nodo_a]

    osm_b = osm_por_id[nodo_b]


    x1 = G.nodes[
        osm_a
    ]["x"]

    y1 = G.nodes[
        osm_a
    ]["y"]


    x2 = G.nodes[
        osm_b
    ]["x"]

    y2 = G.nodes[
        osm_b
    ]["y"]


    # Aproximación sencilla
    # entre dos coordenadas.
    dx = (
        x2 - x1
    ) * 111320


    dy = (
        y2 - y1
    ) * 111320


    return (
        dx ** 2
        +
        dy ** 2
    ) ** 0.5


def ejecutar_bellman_ford(
    inicio,
    final
):

    try:

        ruta = nx.bellman_ford_path(
            H,
            inicio,
            final,
            weight="weight"
        )

    except nx.NetworkXNoPath:

        return []


    return ruta


def ejecutar_floyd_warshall(
    inicio,
    final
):

    try:

        caminos = (
            nx.floyd_warshall_predecessor_and_distance(
                H,
                weight="weight"
            )
        )

        pred = caminos[0]


        ruta = nx.reconstruct_path(
            inicio,
            final,
            pred
        )


    except nx.NetworkXNoPath:

        return []


    except KeyError:

        return []


    return ruta


def ejecutar_algoritmo(
    nombre_algoritmo,
    inicio,
    final
):

    if nombre_algoritmo == "BFS":

        return ejecutar_bfs(
            inicio,
            final
        )

    elif nombre_algoritmo == "DFS":

        return ejecutar_dfs(
            inicio,
            final
        )

    elif nombre_algoritmo == "Dijkstra":

        return ejecutar_dijkstra(
            inicio,
            final
        )

    elif nombre_algoritmo == "A*":

        return ejecutar_a_estrella(
            inicio,
            final
        )

    elif nombre_algoritmo == "Bellman-Ford":

        return ejecutar_bellman_ford(
            inicio,
            final
        )

    else:

        return ejecutar_floyd_warshall(
            inicio,
            final
        )


# ============================================================
# ESTADO DE LA INTERFAZ
# ============================================================

inicio = None

final = None

marcador_inicio = None

marcador_final = None

ruta_dibujada = []


# ============================================================
# DIBUJAR RUTA
# ============================================================

def limpiar_ruta():

    global ruta_dibujada


    for linea in ruta_dibujada:

        linea.remove()

    ruta_dibujada = []

    fig.canvas.draw_idle()


def dibujar_ruta(
    ruta,
    nombre_algoritmo,
    inicio,
    final
):

    global ruta_dibujada

    limpiar_ruta()


    for i in range(
        len(ruta) - 1
    ):

        nodo_a = ruta[i]

        nodo_b = ruta[i + 1]


        osm_a = osm_por_id[nodo_a]

        osm_b = osm_por_id[nodo_b]


        conexiones = G.get_edge_data(
            osm_a,
            osm_b
        )


        if conexiones is None:

            continue


        mejor_key = min(
            conexiones,
            key=lambda k:
            conexiones[k].get(
                "length",
                0
            )
        )


        datos = conexiones[
            mejor_key
        ]


        if "geometry" in datos:

            geometria = datos[
                "geometry"
            ]

            xs, ys = geometria.xy

            linea, = ax.plot(
                xs,
                ys,
                color="yellow",
                linewidth=5,
                zorder=10
            )


        else:

            x1 = G.nodes[
                osm_a
            ]["x"]

            y1 = G.nodes[
                osm_a
            ]["y"]

            x2 = G.nodes[
                osm_b
            ]["x"]

            y2 = G.nodes[
                osm_b
            ]["y"]

            linea, = ax.plot(
                [x1, x2],
                [y1, y2],
                color="yellow",
                linewidth=5,
                zorder=10
            )


        ruta_dibujada.append(linea)


    distancia_total = 0


    for i in range(
        len(ruta) - 1
    ):

        distancia_total += distancia_entre_nodos(
            ruta[i],
            ruta[i + 1]
        )


    ax.set_title(
        f"{nombre_algoritmo} | "
        f"{inicio} → {final} | "
        f"{distancia_total:.2f} m"
    )

    fig.canvas.draw_idle()


# ============================================================
# MARCAR INICIO Y FINAL
# ============================================================

def marcar_inicio(nodo_id):

    global marcador_inicio


    if marcador_inicio is not None:

        marcador_inicio.remove()


    osm = osm_por_id[nodo_id]

    marcador_inicio = ax.scatter(
        G.nodes[osm]["x"],
        G.nodes[osm]["y"],
        s=180,
        color="green",
        edgecolors="black",
        zorder=15
    )

    fig.canvas.draw_idle()


def marcar_final(nodo_id):

    global marcador_final


    if marcador_final is not None:

        marcador_final.remove()


    osm = osm_por_id[nodo_id]

    marcador_final = ax.scatter(
        G.nodes[osm]["x"],
        G.nodes[osm]["y"],
        s=180,
        color="purple",
        edgecolors="black",
        zorder=15
    )

    fig.canvas.draw_idle()


# ============================================================
# SELECCIÓN CON EL CURSOR
# ============================================================

def clic_mapa(event):

    global inicio
    global final


    if event.inaxes is not ax:

        return


    if (
        event.xdata is None
        or
        event.ydata is None
    ):

        return


    if inicio is not None and final is not None:

        return


    _, indice = arbol_nodos.query(
        [event.xdata, event.ydata]
    )

    nodo_osm = lista_nodos[indice]

    nodo_x = G.nodes[
        nodo_osm
    ]["x"]

    nodo_y = G.nodes[
        nodo_osm
    ]["y"]


    distancia_nodo = (
        (
            event.xdata - nodo_x
        ) ** 2
        +
        (
            event.ydata - nodo_y
        ) ** 2
    ) ** 0.5


    if distancia_nodo >= 0.0006:

        return


    nodo_id = id_nuevo[
        nodo_osm
    ]


    if inicio is None:

        inicio = nodo_id

        marcar_inicio(nodo_id)

        ax.set_title(
            f"Inicio: {nodo_id} "
            f"| Haz clic para elegir el FINAL."
        )

    else:

        final = nodo_id

        marcar_final(nodo_id)

        ax.set_title(
            f"Inicio: {inicio} | "
            f"Final: {final} "
            f"| Elige algoritmo y presiona EJECUTAR."
        )


    fig.canvas.draw_idle()


# ============================================================
# BOTONES Y RADIOBUTTONS
# ============================================================

ax_radio = fig.add_axes(
    [0.78, 0.45, 0.2, 0.4]
)

radio_algoritmos = RadioButtons(
    ax_radio,
    (
        "BFS",
        "DFS",
        "Dijkstra",
        "A*",
        "Bellman-Ford",
        "Floyd-Warshall"
    ),
    active=2
)


def al_ejecutar(event):

    if inicio is None or final is None:

        ax.set_title(
            "Primero selecciona INICIO y FINAL en el mapa."
        )

        fig.canvas.draw_idle()

        return


    nombre = radio_algoritmos.value_selected

    ruta = ejecutar_algoritmo(
        nombre,
        inicio,
        final
    )


    if not ruta:

        ax.set_title(
            f"{nombre} | "
            f"No se encontró una ruta entre "
            f"{inicio} y {final}."
        )

        fig.canvas.draw_idle()

        return


    dibujar_ruta(
        ruta,
        nombre,
        inicio,
        final
    )


def al_reiniciar(event):

    global inicio
    global final
    global marcador_inicio
    global marcador_final


    if marcador_inicio is not None:

        marcador_inicio.remove()

        marcador_inicio = None


    if marcador_final is not None:

        marcador_final.remove()

        marcador_final = None


    inicio = None

    final = None

    limpiar_ruta()

    ax.set_title(
        "Haz clic sobre el mapa para elegir el nodo de INICIO, "
        "luego el FINAL."
    )

    fig.canvas.draw_idle()


ax_boton_ejecutar = fig.add_axes(
    [0.78, 0.37, 0.2, 0.06]
)

boton_ejecutar = Button(
    ax_boton_ejecutar,
    "Ejecutar"
)

ax_boton_reiniciar = fig.add_axes(
    [0.78, 0.30, 0.2, 0.06]
)

boton_reiniciar = Button(
    ax_boton_reiniciar,
    "Reiniciar"
)


boton_ejecutar.on_clicked(
    al_ejecutar
)

boton_reiniciar.on_clicked(
    al_reiniciar
)


ax_ayuda = fig.add_axes(
    [0.78, 0.03, 0.2, 0.24]
)

ax_ayuda.axis("off")

ax_ayuda.text(
    0,
    1,
    "INSTRUCCIONES\n\n"
    "1) Clic en el mapa = INICIO\n"
    "2) Clic en el mapa = FINAL\n"
    "3) Elige el algoritmo\n"
    "4) Presiona EJECUTAR\n\n"
    "Reiniciar limpia todo.",
    va="top",
    ha="left",
    fontsize=9
)


# ============================================================
# CONECTAR EVENTOS
# ============================================================

fig.canvas.mpl_connect(
    "button_press_event",
    clic_mapa
)


# ============================================================
# MOSTRAR
# ============================================================

plt.show()
