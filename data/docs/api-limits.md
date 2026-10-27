# API rate limits

The free plan allows 60 requests per minute. The pro plan allows 600 requests per minute, and the enterprise plan has custom limits agreed in the contract.

When a limit is exceeded the API returns HTTP status 429 with a Retry-After header. Clients should back off exponentially and retry after the number of seconds given in that header.
