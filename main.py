import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    tema: str
    datos_busqueda: str
    borrador: str
    revisiones: int
    aprobado: bool
    
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


search_tool = TavilySearch(max_results=3)

def node_investigador(state: AgentState):
    print("NODO: INVESTIGANDO... :)")
    tema = state["tema"]
    
    resultados = search_tool.invoke(tema)
    
    if isinstance(resultados, list):
        textos_encontrados = "\n\n".join([res["content"] for res in resultados])
    else:
        textos_encontrados = str(resultados)
    
    datos_previos = state.get("datos_busqueda", "")
    nuevos_datos = datos_previos + "\n\n--- Nueva Búsqueda ---\n" + textos_encontrados
    
    return {"datos_busqueda": nuevos_datos} 

def node_redactor(state: AgentState):
    print("NODO: Redactor de Borrador ;)")
    tema = state["tema"]
    datos = state.get("datos_busqueda", "")
    
    prompt = f"""
        Eres un redactor experto. Escribe un resumen claro y conciso sobre: '{tema}'.
        Basate ESTRICTAMENTE en la siguiente información encontrada en internet: {datos}
        
        Si la información no es suficiente, haz lo mejor que puedas con lo que tienes.
    """
    
    respuesta = llm.invoke([HumanMessage(content=prompt)])
    
    revisiones_actuales = state.get("revisiones", 0)
    
    return{
        "borrador": respuesta.content,
        "revisiones": revisiones_actuales + 1
    }
    
def node_evaluador(state: AgentState):
    print("NODO: Evaluador de Borrador B)")
    tema = state["tema"]
    borrador = state.get("borrador", "")
    
    prompt = f"""
    Eres un crítico de contenido estricto. 
    El usuario pidió información sobre el tema: '{tema}'
    El borrador actual es: {borrador}
    
    ¿El borrador responde de manera completa y coherente a la solicitud?
    Responde ÚNICAMENTE con la palabra APROBADO o RECHAZADO. No agregues más texto.
    """
    
    respuesta = llm.invoke([HumanMessage(content=prompt)])
    contenido = respuesta.content
    texto_decision = str(contenido).upper()
    
    aprobado = True if "APROBADO" in texto_decision else False
    print(f"   -> Decisión del crítico: {'✅ APROBADO' if aprobado else '❌ RECHAZADO'}")
    
    return {"aprobado": aprobado}

def decidir_siguiente_paso(state: AgentState):
    if state.get("aprobado"):
        return "finalizar"
    
    if state.get("revisiones", 0) >= 3:
        print("Límite de revisiones alcanzado.")
        return "finalizar"
    
    print("Investigan más...")
    return "investigar_mas"

workflow = StateGraph(AgentState)

workflow.add_node("investigador", node_investigador)
workflow.add_node("redactor", node_redactor)
workflow.add_node("evaluador", node_evaluador)

workflow.set_entry_point("investigador")

workflow.add_edge("investigador", "redactor")
workflow.add_edge("redactor", "evaluador")

workflow.add_conditional_edges(
    "evaluador",
    decidir_siguiente_paso,
    {
        "finalizar": END,
        "investigar_mas": "investigador"
    }
)

app = workflow.compile()

if __name__ == "__main__":
    print("IniciandooooooOoOOOoOO agente :D")
    
    estado_inicial = {
        "tema": "¿?",
        "revisiones": 0
    }
    
    resultado_final = app.invoke(estado_inicial)
    
    print("\n" + "="*50)
    print("📄 RESULTADO FINAL APROBADO:")
    print("="*50)
    print(resultado_final["borrador"])