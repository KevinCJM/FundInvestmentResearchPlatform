"""Event-library endpoints installed on the existing historical-regime router."""
from datetime import date
from typing import Literal
from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field
from .event_library import EventDraft, EventSelection


class EventWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: EventDraft


class EventUpdate(EventWrite):
    revision: int = Field(ge=1, strict=True)


class EventResolve(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selections: list[EventSelection] = Field(min_length=1, max_length=100)


class EventImport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition_id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=1, strict=True)


def install_event_routes(router, service, call):
    prefix = "/api/historical-regimes/event-library"

    @router.get(prefix + "/events")
    def events(query: str = Query("", max_length=200), category: str = Query("", max_length=40),
               region: str = Query("", max_length=80), verification: Literal["", "verified", "unreviewed"] = "",
               start: date | None = None, end: date | None = None, archived: bool = False,
               offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
        return call(service().event_library.list, query=query, category=category, region=region,
                    verification=verification, start=start.isoformat() if start else "", end=end.isoformat() if end else "",
                    archived=archived, offset=offset, limit=limit)

    @router.get(prefix + "/events/{event_id}")
    def event(event_id: str, revision: int | None = Query(None, ge=1)):
        return call(service().event_library.get, event_id, revision)

    @router.get(prefix + "/events/{event_id}/history")
    def history(event_id: str):
        return call(service().event_library.history, event_id)

    @router.post(prefix + "/events", status_code=201)
    def create(payload: EventWrite):
        return call(service().event_library.create, payload.event.model_dump(mode="json"))

    @router.put(prefix + "/events/{event_id}")
    def update(event_id: str, payload: EventUpdate):
        return call(service().event_library.update, event_id, payload.revision, payload.event.model_dump(mode="json"))

    @router.get(prefix + "/packs")
    def packs():
        return call(service().event_library.packs)

    @router.get(prefix + "/packs/{pack_id}")
    def pack(pack_id: str):
        return call(service().event_library.pack, pack_id)

    @router.post(prefix + "/resolve")
    def resolve(payload: EventResolve):
        return {"events": call(service().event_library.resolve, [s.model_dump() for s in payload.selections])}

    @router.post(prefix + "/import-definition")
    def import_definition(payload: EventImport):
        definition = call(service().get_definition, payload.definition_id, payload.revision)
        return call(service().event_library.import_definition, definition)
