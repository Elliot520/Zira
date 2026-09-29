// Zira Calendar: a tiny helper app so Zira can read and add Apple Calendar events (2026-09-28).
// Zira's Python can't be given calendar access by macOS (it has no permission description), so this app - which has
// one - asks once ("Zira Calendar would like to access your calendar") and then answers Zira's requests.
// Usage (run through `open -W -g -n -a ZiraCalendar.app --args ...`; the last argument is where the JSON answer goes):
//   status <out>
//   list <start ISO8601> <end ISO8601> <out>
//   add <title> <start ISO8601> <end ISO8601> <notes> <out>
import EventKit
import Foundation

let args = CommandLine.arguments
guard args.count >= 3 else { exit(2) }
let out = URL(fileURLWithPath: args[args.count - 1])
let store = EKEventStore()
let iso = ISO8601DateFormatter()

func answer(_ payload: [String: Any]) -> Never {
    let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
    try? data.write(to: out)
    exit(0)
}

func access() -> (Bool, String) {
    let status = EKEventStore.authorizationStatus(for: .event)
    switch status {
    case .fullAccess, .authorized: return (true, "granted")
    case .denied, .restricted: return (false, "denied")
    case .writeOnly: return (false, "write-only")
    default: break
    }
    let done = DispatchSemaphore(value: 0)
    var granted = false
    store.requestFullAccessToEvents { ok, _ in granted = ok; done.signal() }
    done.wait()
    return (granted, granted ? "granted" : "denied")
}

let (granted, state) = access()
if args[1] == "status" { answer(["ok": true, "access": state]) }
guard granted else { answer(["ok": false, "access": state, "error": "no calendar access (\(state))"]) }

switch args[1] {
case "list" where args.count >= 5:
    guard let start = iso.date(from: args[2]), let end = iso.date(from: args[3]) else {
        answer(["ok": false, "error": "bad dates"])
    }
    let events = store.events(matching: store.predicateForEvents(withStart: start, end: end, calendars: nil))
        .sorted { $0.startDate < $1.startDate }
        .map { e -> [String: Any] in
            ["title": e.title ?? "", "start": iso.string(from: e.startDate), "end": iso.string(from: e.endDate),
             "all_day": e.isAllDay, "location": e.location ?? "", "calendar": e.calendar?.title ?? ""]
        }
    answer(["ok": true, "access": state, "events": events])
case "add" where args.count >= 7:
    guard let start = iso.date(from: args[3]), let end = iso.date(from: args[4]) else {
        answer(["ok": false, "error": "bad dates"])
    }
    let event = EKEvent(eventStore: store)
    event.title = args[2]
    event.startDate = start
    event.endDate = end
    if !args[5].isEmpty { event.notes = args[5] }
    event.calendar = store.defaultCalendarForNewEvents
    do {
        try store.save(event, span: .thisEvent, commit: true)
        answer(["ok": true, "access": state, "calendar": event.calendar?.title ?? ""])
    } catch {
        answer(["ok": false, "error": error.localizedDescription])
    }
default:
    answer(["ok": false, "error": "unknown command"])
}
