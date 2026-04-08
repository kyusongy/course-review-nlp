"""Async httpx GraphQL client for RMP with retry and caching."""

import asyncio
import json
from pathlib import Path

import httpx

RMP_URL = "https://www.ratemyprofessors.com/graphql"
HEADERS = {
    "Authorization": "Basic dGVzdDp0ZXN0",
    "Referer": "https://www.ratemyprofessors.com/",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
SCHOOL_ID = "U2Nob29sLTEyMzI="
TARGET_DEPARTMENTS = {"Statistics", "Statistics & Ops Research", "Biostatistics"}
CACHE_DIR = Path("data/raw")

SEARCH_QUERY = """
query SearchTeachers($schoolID: ID!) {
  newSearch {
    teachers(query: {text: "", schoolID: $schoolID}, first: 500) {
      edges {
        node {
          id legacyId firstName lastName department
          avgRating numRatings avgDifficulty wouldTakeAgainPercent
        }
      }
    }
  }
}
"""

RATINGS_QUERY = """
query GetRatings($id: ID!, $cursor: String) {
  node(id: $id) {
    ... on Teacher {
      id firstName lastName department
      ratings(first: 20, after: $cursor) {
        edges {
          node {
            id comment qualityRating difficultyRating class date
            thumbsUpTotal thumbsDownTotal wouldTakeAgain isForOnlineClass
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


async def _post(client: httpx.AsyncClient, payload: dict, attempts: int = 3) -> dict:
    delay = 1.0
    for attempt in range(attempts):
        try:
            resp = await client.post(RMP_URL, json=payload, headers=HEADERS, timeout=30)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


async def fetch_teachers(client: httpx.AsyncClient) -> list[dict]:
    payload = {"query": SEARCH_QUERY, "variables": {"schoolID": SCHOOL_ID}}
    data = await _post(client, payload)
    edges = data["data"]["newSearch"]["teachers"]["edges"]
    return [e["node"] for e in edges]


async def fetch_ratings_page(
    client: httpx.AsyncClient, teacher_id: str, cursor: str | None = None
) -> dict:
    payload = {
        "query": RATINGS_QUERY,
        "variables": {"id": teacher_id, "cursor": cursor},
    }
    data = await _post(client, payload)
    return data["data"]["node"]["ratings"]


async def fetch_all_ratings(
    client: httpx.AsyncClient, teacher_id: str, legacy_id: int
) -> list[dict]:
    cache_path = CACHE_DIR / f"{legacy_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    ratings = []
    cursor = None
    while True:
        page = await fetch_ratings_page(client, teacher_id, cursor)
        ratings.extend(e["node"] for e in page["edges"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
        await asyncio.sleep(0.5)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(ratings))
    return ratings
