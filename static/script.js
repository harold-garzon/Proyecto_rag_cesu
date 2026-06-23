/* ===================================================================
   Calidoso — lógica del chat
   Compatible con el backend Flask:  POST /chat {message} -> {response, timestamp}
                                     POST /clear
   =================================================================== */

const stream     = document.getElementById("messages");
const entrada    = document.getElementById("message-input");
const btnEnviar  = document.getElementById("send-btn");
const btnLimpiar = document.getElementById("clear-btn");
const contador   = document.getElementById("char-count");

// Marcadores de cita que devuelve el backend (rag_engine -> app.py)
const ICONO = { "📚": "Fuente", "📑": "Artículo", "📄": "Página" };

/* ── utilidades de render ──────────────────────────────────────── */
function escapar(t){ const d = document.createElement("div"); d.textContent = t; return d.innerHTML; }

// Separa el cuerpo de la respuesta de las líneas de cita y las formatea aparte
function construirBurbuja(texto){
  const lineas = texto.split("\n");
  const cuerpo = [], citas = [];

  for (const ln of lineas){
    const ini = ln.trim().slice(0, 2);
    if (ICONO[ini] !== undefined){
      const sinIcono = ln.trim().slice(2);
      const idx = sinIcono.indexOf(":");
      const val = idx >= 0 ? sinIcono.slice(idx + 1).trim() : sinIcono.trim();
      citas.push({ et: ICONO[ini], val });
    } else {
      cuerpo.push(ln);
    }
  }

  // cuerpo → párrafos con **negrita** y saltos de línea
  const parrafos = cuerpo.join("\n").split(/\n{2,}/).filter(p => p.trim() !== "");
  let html = parrafos.map(p => {
    const conSalto = escapar(p).replace(/\n/g, "<br>");
    return "<p>" + conSalto.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>") + "</p>";
  }).join("");
  if (parrafos.length === 0){
    html = "<p>" + escapar(cuerpo.join(" ")).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>") + "</p>";
  }

  // citas → bloque "Referencia normativa"
  if (citas.length){
    let ref = '<div class="ref"><div class="rotulo">Referencia normativa</div>';
    for (const c of citas){
      ref += `<div class="fila"><span class="et">${escapar(c.et)}</span><span class="val">${escapar(c.val)}</span></div>`;
    }
    ref += "</div>";
    html += ref;
  }
  return html;
}

function ahora(){
  const d = new Date();
  return d.getHours().toString().padStart(2,"0") + ":" + d.getMinutes().toString().padStart(2,"0");
}

function agregar(texto, rol, hora){
  const turno = document.createElement("div");
  turno.className = "turno " + rol;

  const av = document.createElement("div");
  av.className = "av " + rol;
  av.textContent = rol === "bot" ? "C" : "Tú";

  const globo = document.createElement("div");
  globo.className = "globo";

  const nombre = document.createElement("div");
  nombre.className = "nombre";
  nombre.textContent = rol === "bot" ? "Calidoso" : "Consulta";

  const burbuja = document.createElement("div");
  burbuja.className = "burbuja";
  if (rol === "bot"){ burbuja.innerHTML = construirBurbuja(texto); }
  else { burbuja.textContent = texto; }

  globo.appendChild(nombre);
  globo.appendChild(burbuja);

  const h = document.createElement("div");
  h.className = "hora";
  h.textContent = hora || ahora();
  globo.appendChild(h);

  turno.appendChild(av);
  turno.appendChild(globo);
  stream.appendChild(turno);
  stream.scrollTop = stream.scrollHeight;
}

function mostrarEscribiendo(){
  const t = document.createElement("div");
  t.className = "turno bot escribiendo";
  t.id = "escribiendo";
  t.innerHTML = '<div class="av bot">C</div><div class="globo"><div class="burbuja"><span class="d"></span><span class="d"></span><span class="d"></span></div></div>';
  stream.appendChild(t);
  stream.scrollTop = stream.scrollHeight;
}
function ocultarEscribiendo(){ const t = document.getElementById("escribiendo"); if (t) t.remove(); }

/* ── envío ─────────────────────────────────────────────────────── */
async function enviar(){
  const texto = entrada.value.trim();
  if (!texto) return;

  agregar(texto, "user");
  entrada.value = "";
  ajustarAlto();
  actualizarContador();
  btnEnviar.disabled = true;
  mostrarEscribiendo();

  try{
    const res  = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: texto })
    });
    const data = await res.json();
    ocultarEscribiendo();
    if (data.error){ agregar("No se pudo procesar la consulta: " + data.error, "bot"); }
    else { agregar(data.response, "bot", data.timestamp); }
  } catch (e){
    ocultarEscribiendo();
    agregar("No hay conexión con el servidor. Verifica que la aplicación esté en ejecución e inténtalo de nuevo.", "bot");
  } finally {
    btnEnviar.disabled = false;
    entrada.focus();
  }
}

async function limpiar(){
  try{ await fetch("/clear", { method: "POST" }); } catch (e){}
  stream.innerHTML = "";
  bienvenida();
  entrada.focus();
}

/* ── interacciones de entrada ──────────────────────────────────── */
function ajustarAlto(){ entrada.style.height = "auto"; entrada.style.height = Math.min(entrada.scrollHeight, 130) + "px"; }
function actualizarContador(){
  const n = entrada.value.length;
  contador.textContent = n + " / 500";
  contador.classList.toggle("alto", n > 450);
}

entrada.addEventListener("input", () => { ajustarAlto(); actualizarContador(); });
entrada.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); enviar(); }
});
btnEnviar.addEventListener("click", enviar);
btnLimpiar.addEventListener("click", limpiar);

/* ── bienvenida ────────────────────────────────────────────────── */
function bienvenida(){
  agregar(
    "Soy **Calidoso**, asistente del Modelo de Acreditación en Alta Calidad del CNA. " +
    "Respondo con base en los documentos normativos cargados y cito la fuente, el artículo y las páginas consultadas en cada respuesta.\n\n" +
    "Escribe tu consulta para comenzar.",
    "bot"
  );
}

bienvenida();
entrada.focus();