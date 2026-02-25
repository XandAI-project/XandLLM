use axum::response::sse::Event;
use serde::Serialize;

/// Serialize a value as a JSON SSE data event.
pub fn sse_event<T: Serialize>(value: &T) -> Result<Event, serde_json::Error> {
    let json = serde_json::to_string(value)?;
    Ok(Event::default().data(json))
}

/// The terminal SSE message that signals the end of a stream.
pub fn sse_done() -> Event {
    Event::default().data("[DONE]")
}
