/** Short PPO facts that rotate on the host beamer during a round (D1). */
export const PPO_FACTS: string[] = [
  "PPO = Proximal Policy Optimization: het 'proximal' slaat op de clip ε die updates klein en veilig houdt.",
  "De clip ε begrenst hoeveel het nieuwe beleid van het oude mag afwijken. Zie het als een lichte 'trust region'.",
  "Een te grote learning rate laat zelfs een goede clip ε ontsporen: beide knoppen bepalen de stapgrootte.",
  "In de praktijk gebruikt PPO vaak ε rond 0.1 tot 0.2 en learning rate rond 3e-4 (Adam).",
  "Stort het beleid in, dan ben je in één update veel van je geleerde gedrag kwijt en herstellen kost lang.",
  "Learning-rate annealing: de stap kleiner maken naarmate de training vordert, voorkomt laat-trainingschaos.",
  "PPO traint on-policy: elke batch ervaring wordt maar een paar epochs hergebruikt, daarna weggegooid.",
  "De sweetspot is geen vast getal. Hij verschuift met de taak en met hoe ver je agent al getraind is.",
];

/** "Wat hebben we geleerd?" recap shown on the podium (D3). */
export const RECAP_POINTS: { title: string; body: string }[] = [
  {
    title: "Clip ε: de trust region",
    body: "Te klein (0.05 of lager) leert traag, ongeveer 0.1 tot 0.3 is de sweetspot, 0.5 of hoger riskeert instorting. Het begrenst hoe ver één update mag gaan.",
  },
  {
    title: "Learning rate: de stapgrootte",
    body: "Te laag kruipt vooruit, te hoog ontspoort. ~3e-4 is een klassieke startwaarde; clip én lr bepalen samen de effectieve stap.",
  },
  {
    title: "De sweetspot verschuift",
    body: "Wat in ronde 1 perfect was, was later te groot. Naarmate de agent rijpt moet je opnieuw afstellen, er is geen magisch getal.",
  },
  {
    title: "Anneal je learning rate",
    body: "Laat in de training werkt een lagere learning rate beter: kleine, precieze stappen in plaats van grote sprongen.",
  },
];

/** One-line credit to the source paper, shown under the recap. */
export const RECAP_SOURCE =
  "Gemodelleerd naar PPO-gedrag uit 'Emergent Autonomous Racing via Multi-Agent PPO' (Sander, MIT).";
