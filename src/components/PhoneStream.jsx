export default function PhoneStream() {
  return (
    <div style={{
      border: "2px solid #333",
      borderRadius: 8,
      overflow: "hidden",
      width: "100%",
      maxWidth: 600,
      background: "#000"
    }}>
      <img
        src="http://127.0.0.1:8000/frame.jpg"
        alt="Phone Stream"
        style={{ width: "100%", display: "block" }}
      />
    </div>
  );
}
