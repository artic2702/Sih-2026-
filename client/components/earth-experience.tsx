'use client'

import React, { memo, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger)
}

/**
 * Checks if WebGL or software WebGL (SwiftShader) is available.
 * Uses failIfMajorPerformanceCaveat: false to allow Chrome environments
 * with disabled GPU hardware acceleration to run WebGL smoothly.
 */
function checkWebGL(): boolean {
  if (typeof window === 'undefined') return false
  try {
    const probe = document.createElement('canvas')
    const gl =
      probe.getContext('webgl2', { failIfMajorPerformanceCaveat: false }) ||
      probe.getContext('webgl', { failIfMajorPerformanceCaveat: false }) ||
      probe.getContext('experimental-webgl', { failIfMajorPerformanceCaveat: false })
    return !!gl
  } catch {
    return false
  }
}

/**
 * Photorealistic 3D Earth Experience
 * Memoized React component to prevent unnecessary re-renders during hero word rotations.
 */
export const EarthExperience = memo(function EarthExperience() {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const earthRef = useRef<THREE.Group | null>(null)
  const cloudsRef = useRef<THREE.Mesh | null>(null)
  const starFieldRef = useRef<THREE.Points | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const scrollTriggerRef = useRef<ScrollTrigger | null>(null)

  const [webglAvailable, setWebglAvailable] = useState(true)

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return

    if (!checkWebGL()) {
      setWebglAvailable(false)
      return
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const width = container.clientWidth || window.innerWidth
    const height = container.clientHeight || window.innerHeight

    // 1. Scene & Perspective Camera
    const scene = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000)
    camera.position.set(0, 0, 5.0)
    cameraRef.current = camera

    // 2. Safe WebGLRenderer with fallback compatibility
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: 'default',
        failIfMajorPerformanceCaveat: false,
      })
      renderer.setSize(width, height, false)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.toneMapping = THREE.ACESFilmicToneMapping
      renderer.toneMappingExposure = 1.35
      rendererRef.current = renderer
    } catch (err) {
      console.warn('WebGLRenderer initialization failed, falling back:', err)
      setWebglAvailable(false)
      return
    }

    // 3. WebGL Context Loss Handlers
    let isContextLost = false
    const handleContextLost = (e: Event) => {
      e.preventDefault()
      isContextLost = true
    }
    const handleContextRestored = () => {
      isContextLost = false
    }
    canvas.addEventListener('webglcontextlost', handleContextLost, false)
    canvas.addEventListener('webglcontextrestored', handleContextRestored, false)

    // 4. Photorealistic Space Lighting
    const ambientLight = new THREE.AmbientLight(0x0a1424, 0.9)
    scene.add(ambientLight)

    // Main Directional Sunlight shining from upper-front-right
    const sunLight = new THREE.DirectionalLight(0xffffff, 4.5)
    sunLight.position.set(3, 8, 6)
    scene.add(sunLight)

    // Blue atmospheric rim light from behind/below for limb scattering
    const rimBacklight = new THREE.DirectionalLight(0x0077ff, 3.0)
    rimBacklight.position.set(0, -1, -5)
    scene.add(rimBacklight)

    // Soft cyan horizon fill
    const horizonFill = new THREE.DirectionalLight(0x38bdf8, 1.0)
    horizonFill.position.set(0, 3, 2)
    scene.add(horizonFill)

    // 5. Earth Group Geometry & Accurate Positioning
    // At camera distance 5.0 and fov 35, vertical span in world units is [-1.58, +1.58]
    // Radius 3.6 with basePosY = -4.02 places the top apex at y = -0.42 (37% from bottom of viewport)
    // Curvature spans across the entire viewport width
    const earthGroup = new THREE.Group()
    const earthRadius = 3.6
    const basePosY = -4.02
    earthGroup.position.set(0, basePosY, 0)

    // Natural 23.5-degree axial tilt + slight forward tilt to reveal equator
    earthGroup.rotation.z = (23.5 * Math.PI) / 180
    earthGroup.rotation.x = 0.22
    // Initial rotation calibrated to display Europe, Mediterranean, Africa, Atlantic Ocean
    earthGroup.rotation.y = 4.2
    scene.add(earthGroup)
    earthRef.current = earthGroup

    // 6. Deep Space Stars (Circular Points shader)
    const starsCount = 1800
    const starGeometry = new THREE.BufferGeometry()
    const starPositions = new Float32Array(starsCount * 3)
    const starColors = new Float32Array(starsCount * 3)
    const starSizes = new Float32Array(starsCount)

    for (let i = 0; i < starsCount; i++) {
      const radius = 95 + Math.random() * 110
      const theta = 2 * Math.PI * Math.random()
      const phi = Math.acos(Math.random() * 1.6 - 0.8)
      const x = radius * Math.sin(phi) * Math.cos(theta)
      const y = radius * Math.sin(phi) * Math.sin(theta)
      const z = radius * Math.cos(phi)

      starPositions[i * 3] = x
      starPositions[i * 3 + 1] = y
      starPositions[i * 3 + 2] = z

      const lum = 0.4 + Math.random() * 0.6
      starColors[i * 3] = lum * 0.88
      starColors[i * 3 + 1] = lum * 0.94
      starColors[i * 3 + 2] = lum

      starSizes[i] = 1.4 + Math.random() * 2.2
    }

    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
    starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3))
    starGeometry.setAttribute('size', new THREE.BufferAttribute(starSizes, 1))

    const starMaterial = new THREE.ShaderMaterial({
      uniforms: {},
      vertexShader: `
        attribute vec3 color;
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (120.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          float dist = length(gl_PointCoord - vec2(0.5));
          if (dist > 0.5) discard;
          float alpha = smoothstep(0.5, 0.05, dist);
          gl_FragColor = vec4(vColor, alpha * 0.85);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })

    const starField = new THREE.Points(starGeometry, starMaterial)
    scene.add(starField)
    starFieldRef.current = starField

    // 7. Earth Surface Mesh & Textures
    const textureLoader = new THREE.TextureLoader()

    const placeholderCanvas = document.createElement('canvas')
    placeholderCanvas.width = 128
    placeholderCanvas.height = 64
    const pCtx = placeholderCanvas.getContext('2d')
    if (pCtx) {
      pCtx.fillStyle = '#08254f'
      pCtx.fillRect(0, 0, 128, 64)
    }
    const placeholderTex = new THREE.CanvasTexture(placeholderCanvas)

    const earthGeometry = new THREE.SphereGeometry(earthRadius, 96, 96)
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: placeholderTex,
      roughness: 0.48,
      metalness: 0.06,
    })
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial)
    earthGroup.add(earthMesh)

    // High-resolution NASA Earth Day Satellite Map
    textureLoader.load(
      '/earth/earth-day.jpg',
      (dayTex) => {
        dayTex.colorSpace = THREE.SRGBColorSpace
        earthMaterial.map = dayTex
        earthMaterial.needsUpdate = true
      },
      undefined,
      (err) => {
        console.error('Failed to load /earth/earth-day.jpg:', err)
        textureLoader.load('/textures/earth/earth-day.jpg', (fallbackDayTex) => {
          fallbackDayTex.colorSpace = THREE.SRGBColorSpace
          earthMaterial.map = fallbackDayTex
          earthMaterial.needsUpdate = true
        })
      }
    )

    // Elevation topology bump map
    textureLoader.load(
      '/earth/earth-topology.png',
      (topoTex) => {
        earthMaterial.bumpMap = topoTex
        earthMaterial.bumpScale = 0.035
        earthMaterial.needsUpdate = true
      },
      undefined,
      (err) => {
        console.error('Failed to load /earth/earth-topology.png:', err)
      }
    )

    // 8. Separate Semi-Transparent Cloud Sphere
    const cloudGeometry = new THREE.SphereGeometry(earthRadius * 1.011, 96, 96)
    const cloudsMaterial = new THREE.MeshStandardMaterial({
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
    const cloudsMesh = new THREE.Mesh(cloudGeometry, cloudsMaterial)
    earthGroup.add(cloudsMesh)
    cloudsRef.current = cloudsMesh

    textureLoader.load(
      '/earth/earth-clouds.png',
      (cloudsTex) => {
        cloudsTex.colorSpace = THREE.SRGBColorSpace
        cloudsMaterial.map = cloudsTex
        cloudsMaterial.needsUpdate = true
      },
      undefined,
      (err) => {
        console.error('Failed to load /earth/earth-clouds.png:', err)
      }
    )

    // 9. Atmospheric Scattering Glow Shader
    const atmosphereGeometry = new THREE.SphereGeometry(earthRadius * 1.018, 96, 96)
    const atmosphereMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uAtmosphereColor: { value: new THREE.Color(0x0088ff) },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vNormal = normalize(mat3(modelMatrix) * normal);
          vViewDir = normalize(cameraPosition - worldPos.xyz);
          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: `
        uniform vec3 uAtmosphereColor;
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          float cosTheta = dot(vNormal, vViewDir);
          float rim = 1.0 - clamp(cosTheta, 0.0, 1.0);
          float crest = pow(rim, 3.8) * 2.8;
          float softHaze = pow(rim, 7.0) * 1.0;
          float alpha = clamp(crest + softHaze, 0.0, 0.95);
          gl_FragColor = vec4(uAtmosphereColor * 1.25, alpha);
        }
      `,
      side: THREE.FrontSide,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
    })
    const atmosphereMesh = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial)
    earthGroup.add(atmosphereMesh)

    // 10. GSAP ScrollTrigger Scrubbed Animation (Smooth, Non-Destructive to DOM)
    const heroSection = document.getElementById('satquery-hero') || container

    if (!prefersReducedMotion) {
      const scrollTl = gsap.timeline({
        scrollTrigger: {
          trigger: heroSection,
          start: 'top top',
          end: 'bottom top',
          scrub: 1.2,
        },
      })

      // As user scrolls down:
      // Earth rises into view, rotates ~100°, camera dollies closer
      scrollTl
        .to(
          earthGroup.position,
          {
            y: basePosY + 1.8,
            z: 0.35,
            ease: 'none',
          },
          0
        )
        .to(
          earthGroup.rotation,
          {
            y: '+=1.8',
            x: 0.12,
            ease: 'none',
          },
          0
        )
        .to(
          camera.position,
          {
            z: 4.4,
            y: 0.15,
            ease: 'none',
          },
          0
        )

      scrollTriggerRef.current = scrollTl.scrollTrigger || null
    }

    // 11. Animation Loop with Continuous Idle Orbital Rotation
    let lastTime = performance.now()

    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate)
      if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return
      if (isContextLost) return

      const now = performance.now()
      const delta = Math.min((now - lastTime) / 1000, 0.1)
      lastTime = now

      if (!prefersReducedMotion && earthRef.current) {
        earthRef.current.rotation.y += delta * 0.012
        if (cloudsRef.current) {
          cloudsRef.current.rotation.y += delta * 0.018
        }
        if (starFieldRef.current) {
          starFieldRef.current.rotation.y += delta * 0.0008
        }
      }

      rendererRef.current.render(sceneRef.current, cameraRef.current)
    }

    animate()

    // 12. Robust Resize Handling
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return
      const w = container.clientWidth || window.innerWidth
      const h = container.clientHeight || window.innerHeight
      if (w === 0 || h === 0) return
      const cam = cameraRef.current
      cam.aspect = w / h
      cam.updateProjectionMatrix()
      rendererRef.current.setSize(w, h, false)
      rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    }

    const resizeObserver = new ResizeObserver(() => handleResize())
    resizeObserver.observe(container)
    window.addEventListener('resize', handleResize)
    handleResize()

    // 13. Comprehensive Lifecycle Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      resizeObserver.disconnect()
      window.removeEventListener('resize', handleResize)
      canvas.removeEventListener('webglcontextlost', handleContextLost)
      canvas.removeEventListener('webglcontextrestored', handleContextRestored)

      if (scrollTriggerRef.current) {
        scrollTriggerRef.current.kill()
      }

      if (rendererRef.current) {
        rendererRef.current.dispose()
      }

      scene.clear()
      earthGeometry.dispose()
      cloudGeometry.dispose()
      atmosphereGeometry.dispose()
      starGeometry.dispose()
      earthMaterial.dispose()
      cloudsMaterial.dispose()
      atmosphereMaterial.dispose()
      starMaterial.dispose()
      placeholderTex.dispose()

      sceneRef.current = null
      cameraRef.current = null
      rendererRef.current = null
      earthRef.current = null
      cloudsRef.current = null
      starFieldRef.current = null
    }
  }, [])

  if (!webglAvailable) {
    return (
      <div
        ref={containerRef}
        className="pointer-events-none absolute inset-0 z-0 flex flex-col justify-end items-center pb-8 overflow-hidden"
        aria-hidden="true"
      >
        <div className="rounded-full border border-white/10 bg-black/40 px-3 py-1 text-xs text-muted-foreground/60 backdrop-blur-sm">
          3D Earth unavailable in this browser
        </div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="pointer-events-none absolute inset-0 z-0 h-full w-full overflow-hidden"
      aria-hidden="true"
    >
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full select-none"
      />
      {/* Subtle bottom gradient transition to blend cleanly into downstream section */}
      <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-28 bg-gradient-to-b from-transparent via-[#0A0A0A]/40 to-[#0A0A0A]" />
    </div>
  )
})
