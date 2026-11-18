export function Placeholder({ title }: { title: string }) {
  return (
    <section aria-label={title}>
      <h2 className="view-title">{title}</h2>
      <p className="muted">Comes in the next slice.</p>
    </section>
  );
}
