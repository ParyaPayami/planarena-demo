# PlanArena: Manhattan web demo

A playable demo of PlanArena, a parcel-level planning model, on 42,075 real Manhattan parcels (NYC DCP MapPLUTO 25v3.1).

**Live:** [planarena-manhattan.onrender.com/play](https://planarena-manhattan.onrender.com/play) (game) and [planarena-manhattan.onrender.com](https://planarena-manhattan.onrender.com) (Studio).

- **Phone game:** `/play`. Re-plan a neighbourhood: add homes, keep parks close, cut emissions, then put your plan to a simulated stakeholder council against the recommended plan.
- **Studio (desktop):** `/`. Paint uses and densities and see 11 indicators with sensitivity ranges, redevelopment likelihood and recommended plans.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ParyaPayami/planarena-demo)

**About this demo.** To fit a free 512 MB server, it runs without PyTorch, so property-value changes use PlanArena's rate-table model instead of the graph-neural-network value head. Everything else is the full model. Impacts are validated in Manhattan only. Votes rest on assumed preferences and are illustrative, not public opinion. The free server sleeps when idle, so the first visit can take about a minute.

Part of the Esri UC 2027 talk *AI Predicts, ArcGIS Pro Proves: Checking Planning Forecasts Against Reality* (Parya Payami, University of Wisconsin–Milwaukee). The ArcGIS Pro checks of PlanArena's forecasts are at [github.com/ParyaPayami/proveit](https://github.com/ParyaPayami/proveit).

**License.** No license has been granted yet: the code is shown to run the demo, and all rights are reserved by the author. PlanArena's full code and paper will be released together.
