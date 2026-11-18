import { useUi } from "../state/store";

export function EmptyState() {
  const setOpenDialog = useUi((s) => s.setOpenDialog);
  return (
    <div className="empty">
      <h2 className="empty-title">Open a vehicle YAML, a saved bundle or an optimizer run to start comparing</h2>
      <button type="button" className="btn primary" onClick={() => setOpenDialog(true)}>
        Open files
      </button>
    </div>
  );
}
