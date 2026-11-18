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

function uploadMultipart(path, file, onProgress, signal, fallbackMessage, onUploadComplete) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const formData = new FormData()
    formData.append('file', file)
    let settled = false

    const cleanup = () => {
      signal?.removeEventListener('abort', abortRequest)
    }
    const finish = (callback, value) => {
      if (settled) return
      settled = true
      cleanup()
      callback(value)
    }
    const abortRequest = () => xhr.abort()

    xhr.open('POST', path)
    xhr.responseType = 'json'
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress?.(Math.min(100, Math.round((event.loaded / event.total) * 100)), event.loaded, event.total)
      }
    }
    xhr.upload.onload = () => onUploadComplete?.()
    xhr.onload = () => {
      const body = xhr.response || {}
      if (xhr.status < 200 || xhr.status >= 300) {
        finish(reject, new Error(body.detail || fallbackMessage))
      } else {
        onProgress?.(100, file.size, file.size)
        finish(resolve, body)
      }
    }
    xhr.onerror = () => finish(reject, new Error('Network error while uploading the file.'))
    xhr.onabort = () => {
      const error = new Error('Upload cancelled.')
      error.name = 'AbortError'
      finish(reject, error)
    }
    if (signal?.aborted) {
      const error = new Error('Upload cancelled.')
      error.name = 'AbortError'
      finish(reject, error)
      return
    }
    signal?.addEventListener('abort', abortRequest, { once: true })
    xhr.send(formData)
  })
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

export async function fetchAutoStoryProjects({ offset = 0, limit = 20, query = '' } = {}) {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) })
  if (query.trim()) params.set('q', query.trim())
  return autoStoryRequest(`/projects?${params.toString()}`)
}

export async function importAutoStoryProject(file, onProgress, signal, onUploadComplete) {
  return uploadMultipart(
    `${API_BASE}/auto-story/projects/import`,
    file,
    onProgress,
    signal,
    'Could not restore Auto Story project',
    onUploadComplete,
  )
}

export async function selectAutoStoryline(jobId, storylineId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/storyline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storyline_id: storylineId }),
  })
}

export async function selectAutoStorylines(jobId, storylineIds, reviewBeforeRender = false) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/storylines/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storyline_ids: storylineIds, review_before_render: reviewBeforeRender }),
  })
}

export async function rediscoverAutoStorylines(jobId, aiProvider) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/storylines/rediscover`, {
    method: 'POST',
    ...(aiProvider ? {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ai_provider: aiProvider }),
    } : {}),
  })
}

export async function saveAutoStorySpeakerNames(jobId, speakerNames, speakerCorrections = [], speakerMerges = {}) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/speakers`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      speaker_names: speakerNames,
      speaker_corrections: speakerCorrections,
      speaker_merges: speakerMerges,
    }),
  })
}

export async function fetchAutoStorySpeakerTranscript(jobId, offset = 0, limit = 100) {
  return autoStoryRequest(
    `/${encodeURIComponent(jobId)}/speakers/transcript?offset=${offset}&limit=${limit}`,
  )
}

export async function retryAutoStoryDiscovery(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/retry-discovery`, {
    method: 'POST',
  })
}

export async function retryAutoStoryline(jobId, storylineId) {
  return autoStoryRequest(
    `/${encodeURIComponent(jobId)}/failed-stories/${encodeURIComponent(storylineId)}/retry`,
    { method: 'POST' },
  )
}

export async function retryAutoStoryRender(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/retry-render`, {
    method: 'POST',
  })
}

export async function reopenAutoStoryReview(jobId) {
  return autoStoryRequest(`/${encodeURIComponent(jobId)}/review-again`, {
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

export async function uploadVideo(file, onProgress, signal) {
  return uploadMultipart(`${API_BASE}/upload`, file, onProgress, signal, 'Upload failed')
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
