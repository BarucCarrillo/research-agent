import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    tema: str
    datos_busqueda: str
    borrador: str
    revisiones: int
    aprobado: bool
    feedback: str
    
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


search_tool = TavilySearch(max_results=3)

def node_investigador(state: AgentState):
    print("NODO: INVESTIGANDO... :)")
    tema = state["tema"]
    feedback = state.get("feedback")
    
    if feedback:
        print(f"Ajustando búsqueda por: {feedback}")
        prompt_busqueda = f"    Tema original: '{tema}'. La búsqueda anterior falló por esto: '{feedback}'. Escribe ÚNICAMENTE una nueva frase de búsqueda corta para Google que resuelva este hueco de información. No escribas nada más."

        respuesta_query = llm.invoke([HumanMessage(content=prompt_busqueda)])
        query_buscar = str(respuesta_query)
        print(f"    Nueva busqueda: {query_buscar}")
    else:
        query_buscar = tema
        print(f"    Búsqueda inicial: {query_buscar}")
        
    resultados = search_tool.invoke(query_buscar)
    
    if isinstance(resultados, list):
        textos_encontrados = "\n\n".join([res["content"] for res in resultados])
    else:
        textos_encontrados = str(resultados)
        
    datos_previos = state.get("datos_busqueda", "")
    nuevos_datos = datos_previos + f"\n\n--- Datos de búsqueda para: {query_buscar} ---\n" + textos_encontrados
        
    return {"datos_busqueda": nuevos_datos} 

def node_redactor(state: AgentState):
    print("NODO: Redactor de Borrador ;)")
    tema = state["tema"]
    datos = state.get("datos_busqueda", "")
    feedback = state.get("feedback", "")
    
    prompt = f"""
        Eres un redactor experto. Escribe un resumen claro y conciso sobre: '{tema}'.
        Basate ESTRICTAMENTE en la siguiente información encontrada en internet: {datos}
        Retroalimentación de la revisión anterior (si existe, asegúrate de cumplirla): {feedback}
        
        Si la información no es suficiente, haz lo mejor que puedas con lo que tienes.
    """
    
    respuesta = llm.invoke([HumanMessage(content=prompt)])
    
    revisiones_actuales = state.get("revisiones", 0)
    
    return{
        "borrador": respuesta.content,
        "revisiones": revisiones_actuales + 1
    }
    
class RespuestaEvaluador(BaseModel):
    aprobado: bool = Field(description="True si el texto es bueno y completo, False si le falta información crucial.")
    feedback: str = Field(description="Si aprobado es False, escribe una instrucción clara sobre qué datos faltan buscar. Si es True, déjalo vacío.")
    
    
def node_evaluador(state: AgentState):
    print("NODO: Evaluador de Borrador B)")
    tema = state["tema"]
    borrador = state.get("borrador", "")
    
    prompt = f"""
    Eres un crítico de contenido estricto. 
    El usuario pidió información sobre el tema: '{tema}'
    El borrador actual es: {borrador}
    
    ¿El borrador responde de manera completa y coherente a la solicitud?
    """
    
    llm_estructurado = llm.with_structured_output(RespuestaEvaluador)
    
    decision = llm_estructurado.invoke([HumanMessage(content=prompt)])
    
    print(f"   -> Decisión del crítico: {'✅ APROBADO' if decision.aprobado else '❌ RECHAZADO'}")
    
    return {"aprobado": decision.aprobado, "feedback": decision.feedback}

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

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    print("IniciandooooooOoOOOoOO agente :D. Escribe tu tema.")
    print("'salir' para salir :P")
    
    config = {"configurable": {"thread_id": "hilo_1"}}
    
    while True:
        tema_usuario = input("q quieres saber???\n>")

        if tema_usuario.lower() in ["salir", "exit", "quit"]:
            print("bye")
            break
        
        if not tema_usuario.strip():
            print("sea serio")
            continue
        
            
        estado_inicial = {
            "tema": tema_usuario,
            "revisiones": 0
        }
        
        print(f"\nIniciando proceso para: '{tema_usuario}'...")
        
        resultado_final = app.invoke(estado_inicial, config=config)
        
        print("\n" + "="*50)
        print("📄 RESULTADO FINAL APROBADO:")
        print("="*50)
        print(resultado_final["borrador"])
        print("="*50 + "\n")