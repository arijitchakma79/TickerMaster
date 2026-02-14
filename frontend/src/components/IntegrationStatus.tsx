interface Props {
  integrations: Record<string, boolean>;
}

export default function IntegrationStatus({ integrations }: Props) {
  const entries = Object.entries(integrations);

  return (
    <section className="integration-row">
      {entries.map(([name, ready]) => (
        <div className="integration-pill" key={name}>
          <span className={ready ? "dot dot-live" : "dot dot-offline"} />
          <span>{name.replace("_", " ")}</span>
        </div>
      ))}
    </section>
  );
}
