import requests
from typing import List, Dict, Optional

BASE_URL = "https://www.googleapis.com/books/v1/volumes"


def _extract_volume_info(item: Dict) -> Dict[str, Optional[str]]:
    info = item.get("volumeInfo", {})
    industry = info.get("industryIdentifiers", [])
    isbn = ""
    for ident in industry:
        if ident.get("type") in ("ISBN_13", "ISBN_10"):
            isbn = ident.get("identifier", isbn)
    thumbnail = ""
    image_links = info.get("imageLinks") or {}
    thumbnail = image_links.get("thumbnail") or image_links.get("smallThumbnail") or ""
    return {
        "title": info.get("title", ""),
        "authors": ", ".join(info.get("authors", [])),
        "categories": ", ".join(info.get("categories", [])),
        "description": info.get("description", ""),
        "publisher": info.get("publisher", ""),
        "published_date": info.get("publishedDate", ""),
        "isbn": isbn,
        "page_count": info.get("pageCount"),
        "thumbnail_url": thumbnail,
    }


def search_by_title(title: str, max_results: int = 5, timeout: int = 5) -> List[Dict[str, Optional[str]]]:
    """Search Google Books by title and return up to `max_results` formatted candidates."""
    params = {
        "q": title,
        "maxResults": max_results,
        "langRestrict": "ja",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        return [_extract_volume_info(it) for it in items]
    except requests.Timeout:
        return []
    except requests.RequestException:
        return []


def search_by_isbn(isbn: str, timeout: int = 5) -> Optional[Dict[str, Optional[str]]]:
    q = f"isbn:{isbn}"
    results = search_by_title(q, max_results=1, timeout=timeout)
    return results[0] if results else None
