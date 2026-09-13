"""Static few-shot estimation examples for CAG context injection."""

from __future__ import annotations

from typing import TypedDict


class EstimationExample(TypedDict):
    meeting_summary: str
    estimation: str


ESTIMATION_EXAMPLES: list[EstimationExample] = [
    {
        "meeting_summary": (
            "El cliente necesita una plataforma web de gestión de inventario "
            "con roles de administrador y operario, CRUD de productos, alertas "
            "de stock bajo e integración con su ERP actual vía API REST. "
            "Diseño pendiente; plazo orientativo de 2 meses."
        ),
        "estimation": """## Estimación: Plataforma de Gestión de Inventario

### Resumen del proyecto
Aplicación web para control de stock con autenticación por roles, CRUD de
productos, alertas y conector REST hacia el ERP existente.

### Desglose de tareas:
1. Diseño UI/UX: 40 horas a 400 EUR/hora → 16.000 EUR
2. Backend API (CRUD inventario): 60 horas a 500 EUR/hora → 30.000 EUR
3. Autenticación y roles: 20 horas a 500 EUR/hora → 10.000 EUR
4. Dashboard con métricas: 30 horas a 500 EUR/hora → 15.000 EUR
5. Integración ERP (API REST): 35 horas a 500 EUR/hora → 17.500 EUR
6. Testing y QA: 25 horas a 450 EUR/hora → 11.250 EUR

**Total estimado: 210 horas / 99.750 EUR**
**Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**

### Riesgos / supuestos
- El ERP expone endpoints documentados y estables.
- No se incluye migración histórica de datos legacy.
""",
    },
    {
        "meeting_summary": (
            "Startup B2B quiere un MVP de SaaS de facturación recurrente: "
            "onboarding de clientes, planes mensuales/anuales, pasarela Stripe, "
            "panel de facturas PDF y webhooks de pago. Diseño ya existe en Figma. "
            "Quieren lanzar en 5 semanas con un equipo pequeño."
        ),
        "estimation": """## Estimación: MVP SaaS de Facturación Recurrente

### Resumen del proyecto
MVP multi-tenant de suscripciones con Stripe, emisión de facturas PDF y
panel de administración básico. UI basada en Figma existente.

### Desglose de tareas:
1. Setup multi-tenant y auth: 25 horas a 500 EUR/hora → 12.500 EUR
2. Catálogo de planes y checkout Stripe: 40 horas a 500 EUR/hora → 20.000 EUR
3. Webhooks de pago y reconciliación: 30 horas a 500 EUR/hora → 15.000 EUR
4. Generación de facturas PDF: 20 horas a 500 EUR/hora → 10.000 EUR
5. Implementación UI desde Figma: 35 horas a 450 EUR/hora → 15.750 EUR
6. Testing e2e de pagos (sandbox): 20 horas a 450 EUR/hora → 9.000 EUR

**Total estimado: 170 horas / 82.250 EUR**
**Equipo recomendado: 2 desarrolladores full-stack**
**Duración estimada: 5 semanas**

### Riesgos / supuestos
- Stripe Checkout + Customer Portal cubren el flujo sin facturación custom compleja.
- No se incluyen impuestos multi-país ni contabilidad avanzada.
""",
    },
    {
        "meeting_summary": (
            "Cadena de restaurantes pide una app móvil (iOS/Android) de reservas "
            "online con calendario de mesas, recordatorios por SMS/email y "
            "panel web para el gerente. Integración con su CRM HubSpot. "
            "Quieren piloto en 3 locales en 10 semanas."
        ),
        "estimation": """## Estimación: App de Reservas para Restaurantes

### Resumen del proyecto
App móvil cross-platform de reservas, panel web de gestión y sincronización
con HubSpot para el seguimiento comercial.

### Desglose de tareas:
1. Diseño UX flujos reserva/gerente: 45 horas a 400 EUR/hora → 18.000 EUR
2. Backend reservas + calendario: 70 horas a 500 EUR/hora → 35.000 EUR
3. App móvil (React Native / Flutter): 90 horas a 500 EUR/hora → 45.000 EUR
4. Panel web gerente: 40 horas a 500 EUR/hora → 20.000 EUR
5. Notificaciones SMS/email: 25 horas a 500 EUR/hora → 12.500 EUR
6. Integración HubSpot: 30 horas a 500 EUR/hora → 15.000 EUR
7. Testing y piloto 3 locales: 35 horas a 450 EUR/hora → 15.750 EUR

**Total estimado: 335 horas / 161.250 EUR**
**Equipo recomendado: 2 mobile + 1 backend + 1 UX (part-time)**
**Duración estimada: 9-10 semanas**

### Riesgos / supuestos
- Un único menú de mesas por local; sin waitlist avanzada en el MVP.
- Costes de SMS se facturan aparte según volumen.
""",
    },
]
