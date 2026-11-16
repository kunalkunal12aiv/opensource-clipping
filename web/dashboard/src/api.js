const API_BASE = '/api'

export async function fetchJobs() {
  const res = await fetch(`${API_BASE}/jobs`)
  if (!res.ok) throw new Error('Failed to fetch jobs')
  return res.json()
}

export async function fetchJob(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`)
  if (!res.ok) throw new Error('Job not found')
  return res.json()
}

export async function createJob(payload) {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to create job')
  }
  return res.json()
}

async function autoStoryRequest(path, options = {}) {
  const res = await fetch(`${API_BASE}/auto-story${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Auto Story request failed')
  }
  return res.json()
}

export async function createAutoStory(payload) {
  return autoStoryRequest('', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function fetchAutoStory(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}`)
}

export async function selectAutoStoryline(jobId, storylineId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/storyline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storyline_id: storylineId }),
  })
}

export async function selectAutoStorylines(jobId, storylineIds) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/storylines/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storyline_ids: storylineIds }),
  })
}

export async function rediscoverAutoStorylines(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/storylines/rediscover`, {
    method: 'POST',
  })
}

export async function validateAutoStoryRecipe(jobId, recipe) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/recipe`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ recipe }),
  })
}

export async function validateAutoStoryPlan(jobId, stories) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/recipe`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stories }),
  })
}

export async function reconstructAutoStoryStory(jobId, storyId) {
  return autoStoryRequest(
    `/${encodeURIComponent(jobId)}/stories/${encodeURIComponent(storyId)}/reconstruct`,
    { method: 'POST' },
  )
}

export async function renderAutoStory(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/render`, {
    method: 'POST',
  })
}

export async function cancelAutoStory(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
  })
}

export async function deleteJob(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete job')
  return res.json()
}

export async function uploadVideo(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}

export async function fetchSettings() {
  const res = await fetch(`${API_BASE}/settings`)
  if (!res.ok) throw new Error('Failed to fetch settings')
  return res.json()
}

export async function updateSettings(payload) {
  const res = await fetch(`${API_BASE}/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Failed to update settings')
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export function createSSEConnection(jobId, onMessage) {
  const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/status`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('SSE parse error:', e)
    }
  }

  eventSource.onerror = () => {
    eventSource.close()
  }

  return eventSource
}
