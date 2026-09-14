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

### Desglose de tareas

| Tarea | Horas | Coste (EUR) |
|------|------:|------------:|
| Diseño UI/UX | 40 | 2.000 |
| Backend API (CRUD inventario) | 56 | 3.500 |
| Autenticación y roles | 16 | 1.000 |
| Dashboard con métricas | 24 | 1.500 |
| Integración ERP (API REST) | 32 | 2.000 |
| Testing y QA | 24 | 1.500 |

### Totales
- **Total horas:** 192
- **Total coste:** 11.500 EUR

### Equipo recomendado
- 2 desarrolladores full-stack
- 1 diseñador UX (part-time)

### Duración estimada
**7-8 semanas**
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

### Desglose de tareas

| Tarea | Horas | Coste (EUR) |
|------|------:|------------:|
| Setup multi-tenant y autenticación | 24 | 1.500 |
| Catálogo de planes y checkout Stripe | 40 | 2.500 |
| Webhooks de pago y reconciliación | 32 | 2.000 |
| Generación de facturas PDF | 16 | 1.000 |
| Implementación UI desde Figma | 32 | 2.000 |
| Testing e2e de pagos (sandbox) | 16 | 1.000 |

### Totales
- **Total horas:** 160
- **Total coste:** 10.000 EUR

### Equipo recomendado
- 2 desarrolladores full-stack

### Duración estimada
**5 semanas**
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

### Desglose de tareas

| Tarea | Horas | Coste (EUR) |
|------|------:|------------:|
| Diseño UX flujos reserva/gerente | 40 | 2.000 |
| Backend reservas y calendario | 64 | 4.000 |
| App móvil (React Native / Flutter) | 80 | 5.000 |
| Panel web gerente | 40 | 2.500 |
| Notificaciones SMS/email | 24 | 1.500 |
| Integración HubSpot | 32 | 2.000 |
| Testing y piloto 3 locales | 32 | 2.000 |

### Totales
- **Total horas:** 312
- **Total coste:** 19.000 EUR

### Equipo recomendado
- 2 desarrolladores mobile
- 1 desarrollador backend
- 1 diseñador UX (part-time)

### Duración estimada
**9-10 semanas**
""",
    },
]
