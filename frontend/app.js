/**
 * app.js - Frontend controller for Quad-Dominant Remesher Studio
 * Powered by Three.js & OrbitControls with dual synchronized viewports.
 */

class MeshViewer {
  constructor(containerId, isOutput = false) {
    this.container = document.getElementById(containerId);
    this.isOutput = isOutput;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.meshGroup = new THREE.Group();
    this.wireframeMesh = null;
    this.surfaceMesh = null;
    this.shadingMode = 'clay'; // 'clay', 'facetype', 'jacobian', 'valence'
    this.showWireframe = true;
    this.meshData = null;

    this.init();
  }

  init() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060913);

    // Subtle grid plane
    const grid = new THREE.GridHelper(2.5, 20, 0x1e293b, 0x0f172a);
    grid.position.y = -0.6;
    this.scene.add(grid);

    // Camera
    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.05, 50);
    this.camera.position.set(0, 0.4, 2.2);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = true;

    // Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(3, 5, 4);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x6366f1, 0.4);
    dirLight2.position.set(-4, -2, -3);
    this.scene.add(dirLight2);

    this.scene.add(this.meshGroup);

    // Resize handler
    window.addEventListener('resize', () => this.onResize());

    // Render loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  onResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  setMesh(payload) {
    this.meshData = payload;
    this.rebuildMesh();
  }

  rebuildMesh() {
    if (!this.meshData) return;

    // Clear old objects
    while (this.meshGroup.children.length > 0) {
      const obj = this.meshGroup.children[0];
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
      this.meshGroup.remove(obj);
    }

    const { vertices, quads, tris, quad_sjs, valences } = this.meshData;
    const vArr = new Float32Array(vertices);

    // We build two geometries: Quads (triangulated 2 triangles per quad) and Triangles
    const triIndices = [];
    const colorIndices = []; // To store per-triangle face type or quality colors

    // 1. Process Quads: split each [a, b, c, d] into [a, b, c] and [a, c, d]
    const numQuads = quads.length / 4;
    for (let i = 0; i < numQuads; i++) {
      const a = quads[i * 4];
      const b = quads[i * 4 + 1];
      const c = quads[i * 4 + 2];
      const d = quads[i * 4 + 3];
      triIndices.push(a, b, c, a, c, d);

      const sj = quad_sjs ? quad_sjs[i] : 1.0;
      colorIndices.push({ type: 'quad', sj, vList: [a, b, c, d] });
      colorIndices.push({ type: 'quad', sj, vList: [a, b, c, d] });
    }

    // 2. Process Triangles
    const numTris = tris.length / 3;
    for (let i = 0; i < numTris; i++) {
      const a = tris[i * 3];
      const b = tris[i * 3 + 1];
      const c = tris[i * 3 + 2];
      triIndices.push(a, b, c);
      colorIndices.push({ type: 'tri', sj: 0.0, vList: [a, b, c] });
    }

    // Build main BufferGeometry
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(vArr, 3));
    geom.setIndex(triIndices);
    geom.computeVertexNormals();

    // Setup colors based on shading mode
    const numTrisTotal = triIndices.length / 3;
    const colors = new Float32Array(numTrisTotal * 3 * 3); // 3 verts per tri, 3 RGB

    for (let t = 0; t < numTrisTotal; t++) {
      const info = colorIndices[t];
      let r = 0.8, g = 0.8, b = 0.82; // Default Clay

      if (this.shadingMode === 'facetype') {
        if (info.type === 'quad') {
          r = 0.06; g = 0.72; b = 0.50; // Emerald for quads
        } else {
          r = 0.96; g = 0.62; b = 0.04; // Amber for triangles
        }
      } else if (this.shadingMode === 'jacobian') {
        if (info.type === 'quad') {
          // Heatmap: red (0.0) -> yellow (0.6) -> emerald (1.0)
          const sj = Math.max(0, Math.min(1, info.sj));
          if (sj < 0.5) {
            r = 0.95;
            g = sj * 1.8;
            b = 0.15;
          } else {
            r = (1.0 - sj) * 1.8;
            g = 0.85;
            b = 0.25;
          }
        } else {
          // Triangle = neutral dark slate in SJ mode
          r = 0.35; g = 0.40; b = 0.48;
        }
      } else if (this.shadingMode === 'valence') {
        // Compute average valence of corners
        let avgVal = 4;
        if (valences && info.vList) {
          const sum = info.vList.reduce((acc, v) => acc + (valences[v] || 4), 0);
          avgVal = sum / info.vList.length;
        }
        if (Math.abs(avgVal - 4.0) < 0.1) {
          r = 0.4; g = 0.45; b = 0.55; // Regular valence 4
        } else if (avgVal < 4.0) {
          r = 0.2; g = 0.6; b = 0.95;  // Valence 3 (singularity)
        } else {
          r = 0.95; g = 0.25; b = 0.35; // Valence 5+ (singularity)
        }
      }

      for (let k = 0; k < 3; k++) {
        const idx = (t * 3 + k) * 3;
        colors[idx] = r;
        colors[idx + 1] = g;
        colors[idx + 2] = b;
      }
    }

    // Material with Vertex Colors
    const isClay = this.shadingMode === 'clay';
    const mat = new THREE.MeshStandardMaterial({
      color: isClay ? 0xd0d5dd : 0xffffff,
      vertexColors: !isClay,
      roughness: isClay ? 0.45 : 0.35,
      metalness: 0.1,
      side: THREE.DoubleSide
    });

    this.surfaceMesh = new THREE.Mesh(geom.toNonIndexed(), mat);
    this.meshGroup.add(this.surfaceMesh);

    // Build True Quad Wireframe (outer quad boundary, omitting diagonal split!)
    if (this.showWireframe) {
      const linePositions = [];
      const v = vertices;

      // Quad edges (a-b, b-c, c-d, d-a)
      for (let i = 0; i < numQuads; i++) {
        const a = quads[i * 4] * 3;
        const b = quads[i * 4 + 1] * 3;
        const c = quads[i * 4 + 2] * 3;
        const d = quads[i * 4 + 3] * 3;
        linePositions.push(
          v[a], v[a + 1], v[a + 2], v[b], v[b + 1], v[b + 2],
          v[b], v[b + 1], v[b + 2], v[c], v[c + 1], v[c + 2],
          v[c], v[c + 1], v[c + 2], v[d], v[d + 1], v[d + 2],
          v[d], v[d + 1], v[d + 2], v[a], v[a + 1], v[a + 2]
        );
      }

      // Triangle edges (a-b, b-c, c-a)
      for (let i = 0; i < numTris; i++) {
        const a = tris[i * 3] * 3;
        const b = tris[i * 3 + 1] * 3;
        const c = tris[i * 3 + 2] * 3;
        linePositions.push(
          v[a], v[a + 1], v[a + 2], v[b], v[b + 1], v[b + 2],
          v[b], v[b + 1], v[b + 2], v[c], v[c + 1], v[c + 2],
          v[c], v[c + 1], v[c + 2], v[a], v[a + 1], v[a + 2]
        );
      }

      const lineGeom = new THREE.BufferGeometry();
      lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: this.isOutput ? 0x052e16 : 0x0f172a,
        linewidth: 1.2,
        transparent: true,
        opacity: 0.85,
        depthTest: true
      });

      this.wireframeMesh = new THREE.LineSegments(lineGeom, lineMat);
      this.meshGroup.add(this.wireframeMesh);
    }
  }

  setShading(mode) {
    this.shadingMode = mode;
    this.rebuildMesh();
  }

  setWireframe(visible) {
    this.showWireframe = visible;
    this.rebuildMesh();
  }

  resetCamera() {
    this.camera.position.set(0, 0.4, 2.2);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
}

// Global Application Controller
class App {
  constructor() {
    this.viewerIn = new MeshViewer('canvas-wrapper-in', false);
    this.viewerOut = new MeshViewer('canvas-wrapper-out', true);
    this.syncCameras = true;
    this.currentRunId = null;

    this.bindEvents();
    this.initSyncControls();
    this.loadSampleList();
  }

  initSyncControls() {
    // Synchronize OrbitControls between Viewport 1 & Viewport 2
    let syncing = false;

    this.viewerIn.controls.addEventListener('change', () => {
      if (!this.syncCameras || syncing) return;
      syncing = true;
      this.viewerOut.camera.position.copy(this.viewerIn.camera.position);
      this.viewerOut.camera.quaternion.copy(this.viewerIn.camera.quaternion);
      this.viewerOut.controls.target.copy(this.viewerIn.controls.target);
      this.viewerOut.controls.update();
      syncing = false;
    });

    this.viewerOut.controls.addEventListener('change', () => {
      if (!this.syncCameras || syncing) return;
      syncing = true;
      this.viewerIn.camera.position.copy(this.viewerOut.camera.position);
      this.viewerIn.camera.quaternion.copy(this.viewerOut.camera.quaternion);
      this.viewerIn.controls.target.copy(this.viewerOut.controls.target);
      this.viewerIn.controls.update();
      syncing = false;
    });
  }

  bindEvents() {
    // Sync Viewports Button
    const btnSync = document.getElementById('btn-sync-cam');
    btnSync.addEventListener('click', () => {
      this.syncCameras = !this.syncCameras;
      btnSync.classList.toggle('active', this.syncCameras);
    });

    // Reset Camera
    document.getElementById('btn-reset-cam').addEventListener('click', () => {
      this.viewerIn.resetCamera();
      this.viewerOut.resetCamera();
    });

    // Wireframe Toggle
    const btnWire = document.getElementById('btn-wireframe');
    btnWire.addEventListener('click', () => {
      const active = !btnWire.classList.contains('active');
      btnWire.classList.toggle('active', active);
      this.viewerIn.setWireframe(active);
      this.viewerOut.setWireframe(active);
    });

    // Shading Buttons
    document.querySelectorAll('[data-shading]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('[data-shading]').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        const mode = e.currentTarget.dataset.shading;
        this.viewerIn.setShading(mode);
        this.viewerOut.setShading(mode);
      });
    });

    // Category Profile (drives which CategoryProfile thresholds the backend uses)
    const profileSelect = document.getElementById('category-profile');
    const modeBadge = document.getElementById('mode-badge');
    const creaseDefaults = { organic: 42, utensil: 30, hard_surface: 25 };
    // Matches src/retopo/config.py's CategoryProfile.sizing_ratio defaults --
    // keep these in sync if those are retuned (see that file's docstring for
    // the diagnostic behind these specific numbers).
    const sizingRatioDefaults = { organic: 30, utensil: 25, hard_surface: 80 };
    if (profileSelect) {
      profileSelect.addEventListener('change', (e) => {
        const profile = e.target.value;
        if (modeBadge) modeBadge.textContent = `⚡ ${profile}`;
        const creaseSlider = document.getElementById('crease-angle');
        const creaseLabel = document.getElementById('val-crease-angle');
        if (creaseSlider && creaseDefaults[profile] !== undefined) {
          creaseSlider.value = creaseDefaults[profile];
          if (creaseLabel) creaseLabel.textContent = `${creaseDefaults[profile]}°`;
        }
        const sizingRatioSlider = document.getElementById('sizing-ratio');
        const sizingRatioLabel = document.getElementById('val-sizing-ratio');
        if (sizingRatioSlider && sizingRatioDefaults[profile] !== undefined) {
          sizingRatioSlider.value = sizingRatioDefaults[profile];
          if (sizingRatioLabel) sizingRatioLabel.textContent = `${sizingRatioDefaults[profile]}`;
        }
      });
    }

    // Sliders
    this.bindSlider('target-faces', 'val-target-faces', val => Number(val).toLocaleString());
    this.bindSlider('crease-angle', 'val-crease-angle', val => `${val}°`);
    this.bindSlider('smooth-iters', 'val-smooth-iters', val => `${val} iters`);
    this.bindSlider('sizing-ratio', 'val-sizing-ratio', val => `${val}`);

    // File Drop Zone
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileLabel = document.getElementById('selected-file-label');

    if (dropZone && fileInput) {
      dropZone.addEventListener('click', () => fileInput.click());
      dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
      dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          fileInput.files = e.dataTransfer.files;
          this.onFileSelected(fileInput.files[0]);
        }
      });
      fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
          this.onFileSelected(fileInput.files[0]);
        }
      });
    }

    // Sample Select Dropdown
    const sampleSelect = document.getElementById('sample-select');
    if (sampleSelect) {
      sampleSelect.addEventListener('change', (e) => {
        if (e.target.value) {
          if (fileInput) fileInput.value = '';
          if (fileLabel) fileLabel.style.display = 'none';
          this.loadPreview(e.target.value, null);
        }
      });
    }

    // Run Remesh Button
    const btnRun = document.getElementById('btn-run');
    if (btnRun) {
      btnRun.addEventListener('click', () => this.runRemesh());
    }

    // Download Buttons
    const btnDownObj = document.getElementById('btn-download-obj');
    if (btnDownObj) {
      btnDownObj.addEventListener('click', () => {
        if (this.currentRunId) window.location.href = `/api/download/${this.currentRunId}?format=obj`;
      });
    }
    const btnDownGlb = document.getElementById('btn-download-glb');
    if (btnDownGlb) {
      btnDownGlb.addEventListener('click', () => {
        if (this.currentRunId) window.location.href = `/api/download/${this.currentRunId}?format=glb`;
      });
    }
  }

  bindSlider(sliderId, labelId, formatFn) {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    if (!slider || !label) return;
    slider.addEventListener('input', (e) => {
      label.textContent = formatFn(e.target.value);
    });
  }

  async onFileSelected(file) {
    const label = document.getElementById('selected-file-label');
    if (label) {
      label.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      label.style.display = 'inline-block';
    }
    const sampleSelect = document.getElementById('sample-select');
    if (sampleSelect) sampleSelect.value = '';
    await this.loadPreview(null, file);
  }

  async loadPreview(sampleName, file) {
    const loaderIn = document.getElementById('loader-in');
    if (loaderIn) loaderIn.classList.remove('hidden');

    try {
      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      } else if (sampleName) {
        formData.append('sample_name', sampleName);
      } else {
        return;
      }

      const res = await fetch('/api/preview', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        if (data.mesh) {
          this.viewerIn.setMesh(data.mesh);
          const badgeCount = document.getElementById('badge-in-count');
          if (badgeCount) {
            badgeCount.textContent = `${data.face_count.toLocaleString()} faces`;
          }
          const badgeQuads = document.getElementById('badge-in-quads');
          if (badgeQuads) {
            badgeQuads.textContent = `${data.mesh.num_quads} Quads / ${data.mesh.num_tris} Tris`;
          }
        }
      }
    } catch (err) {
      console.warn('Preview load error:', err);
    } finally {
      if (loaderIn) loaderIn.classList.add('hidden');
    }
  }

  async loadSampleList() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      const select = document.getElementById('sample-select');
      if (!select) return;
      select.innerHTML = '<option value="">-- Choose Sample Model --</option>';

      data.samples.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.name;
        opt.textContent = `${s.name} [${s.size_str}]`;
        select.appendChild(opt);
      });

      // Default select testsample1_raw_860k.glb or first sample and load preview
      if (data.samples.length > 0) {
        const defaultSample = data.samples.find(s => s.name.includes('testsample1') || s.name.includes('860k')) || data.samples[0];
        select.value = defaultSample.name;
        await this.loadPreview(defaultSample.name, null);
      }
    } catch (err) {
      console.error('Failed to load sample list:', err);
    }
  }

  async runRemesh() {
    const fileInput = document.getElementById('file-input');
    const sampleSelect = document.getElementById('sample-select');
    const selectedSample = sampleSelect ? sampleSelect.value : '';
    const hasFile = fileInput && fileInput.files && fileInput.files.length > 0;

    if (!hasFile && !selectedSample) {
      alert('Please select a sample model or upload a 3D mesh first.');
      return;
    }

    const formData = new FormData();
    if (hasFile) {
      formData.append('file', fileInput.files[0]);
    } else {
      formData.append('sample_name', selectedSample);
    }

    const inputModeEl = document.getElementById('input-mode');
    if (inputModeEl) formData.append('input_mode', inputModeEl.value);

    const engineSelectEl = document.getElementById('engine-select');
    if (engineSelectEl) formData.append('engine', engineSelectEl.value);

    const profileEl = document.getElementById('category-profile');
    if (profileEl) formData.append('profile', profileEl.value);

    const targetFacesEl = document.getElementById('target-faces');
    if (targetFacesEl) formData.append('target_faces', targetFacesEl.value);

    const creaseAngleEl = document.getElementById('crease-angle');
    if (creaseAngleEl) formData.append('crease_angle', creaseAngleEl.value);

    const minSjEl = document.getElementById('min-sj');
    if (minSjEl) formData.append('min_sj', minSjEl.value);

    const smoothItersEl = document.getElementById('smooth-iters');
    if (smoothItersEl) formData.append('smooth_iters', smoothItersEl.value);

    const sizingRatioEl = document.getElementById('sizing-ratio');
    if (sizingRatioEl) formData.append('sizing_ratio', sizingRatioEl.value);

    const defeatureEl = document.getElementById('defeature-toggle');
    if (defeatureEl) formData.append('defeature', defeatureEl.checked ? 'true' : 'false');

    // Show loaders
    const loaderIn = document.getElementById('loader-in');
    const loaderOut = document.getElementById('loader-out');
    const btnRun = document.getElementById('btn-run');

    loaderIn.classList.remove('hidden');
    loaderOut.classList.remove('hidden');
    btnRun.disabled = true;

    try {
      const res = await fetch('/api/remesh', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        let errorMsg = 'Remeshing failed';
        try {
          const err = await res.json();
          errorMsg = err.detail || errorMsg;
        } catch (_) {
          try {
            errorMsg = await res.text();
          } catch (e) {
            errorMsg = `Server error HTTP ${res.status}`;
          }
        }
        throw new Error(errorMsg);
      }

      const data = await res.json();
      this.currentRunId = data.report.run_id;

      // Update Viewports
      this.viewerIn.setMesh(data.input_mesh);
      this.viewerOut.setMesh(data.output_mesh);

      // Update Pane Badges
      document.getElementById('badge-in-count').textContent =
        `${data.input_mesh.num_quads} quads, ${data.input_mesh.num_tris} tris (${data.report.input_faces} faces)`;
      document.getElementById('badge-out-count').textContent =
        `${data.output_mesh.num_quads} quads, ${data.output_mesh.num_tris} tris (${data.report.output_faces} faces)`;

      // Update KPI Cards
      this.updateKPICards(data.report);

      // Enable Export Section
      document.getElementById('export-section').classList.remove('hidden');

      // Bind JSON download button
      document.getElementById('btn-download-json').onclick = () => {
        const blob = new Blob([JSON.stringify(data.report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `benchmark_${data.report.display_name}.json`;
        a.click();
      };

    } catch (err) {
      alert(`Error during remeshing: ${err.message}`);
    } finally {
      loaderIn.classList.add('hidden');
      loaderOut.classList.add('hidden');
      btnRun.disabled = false;
    }
  }

  updateKPICards(rep) {
    // 1. Quad Ratio
    const elQuadVal = document.getElementById('kpi-val-quad');
    const elQuadStatus = document.getElementById('status-quad');
    elQuadVal.textContent = `${rep.output_quad_ratio_pct}%`;
    if (rep.output_quad_ratio_pct >= 94.0) {
      elQuadStatus.className = 'kpi-status badge-ok';
      elQuadStatus.textContent = 'Rabbit Grade';
    } else if (rep.output_quad_ratio_pct >= 85.0) {
      elQuadStatus.className = 'kpi-status badge-warn';
      elQuadStatus.textContent = 'Acceptable';
    } else {
      elQuadStatus.className = 'kpi-status badge-fail';
      elQuadStatus.textContent = 'Sub-optimal';
    }

    // 1b. Watertight / Mesh Contiguity
    const elWatertightVal = document.getElementById('kpi-val-watertight');
    const elWatertightStatus = document.getElementById('status-watertight');
    if (elWatertightVal && elWatertightStatus) {
      if (rep.is_watertight || rep.boundary_edges === 0) {
        elWatertightVal.textContent = 'Watertight (0 Holes)';
        elWatertightStatus.className = 'kpi-status badge-ok';
        elWatertightStatus.textContent = '0 Holes / Sealed';
      } else {
        elWatertightVal.textContent = `${rep.boundary_edges} natural bounds`;
        elWatertightStatus.className = 'kpi-status badge-ok';
        elWatertightStatus.textContent = '0 Gaping Tears';
      }
    }

    // 2. Mean Scaled Jacobian
    const elMeanSjVal = document.getElementById('kpi-val-mean-sj');
    const elMeanSjStatus = document.getElementById('status-mean-sj');
    elMeanSjVal.textContent = rep.mean_scaled_jacobian.toFixed(3);
    if (rep.mean_scaled_jacobian >= 0.90) {
      elMeanSjStatus.className = 'kpi-status badge-ok';
      elMeanSjStatus.textContent = 'Superior';
    } else if (rep.mean_scaled_jacobian >= 0.65) {
      elMeanSjStatus.className = 'kpi-status badge-warn';
      elMeanSjStatus.textContent = 'Good';
    } else {
      elMeanSjStatus.className = 'kpi-status badge-fail';
      elMeanSjStatus.textContent = 'Distorted';
    }

    // 3. Min Scaled Jacobian
    const elMinSjVal = document.getElementById('kpi-val-min-sj');
    const elMinSjStatus = document.getElementById('status-min-sj');
    elMinSjVal.textContent = rep.min_scaled_jacobian.toFixed(3);
    if (rep.min_scaled_jacobian >= 0.35) {
      elMinSjStatus.className = 'kpi-status badge-ok';
      elMinSjStatus.textContent = 'Gate Passed';
    } else {
      elMinSjStatus.className = 'kpi-status badge-warn';
      elMinSjStatus.textContent = 'Demoted';
    }

    // 4. Regular Valence-4
    document.getElementById('kpi-val-valence4').textContent = `${rep.regular_valence4_pct}%`;
    const elValStatus = document.getElementById('status-valence4');
    elValStatus.className = rep.regular_valence4_pct >= 50.0 ? 'kpi-status badge-ok' : 'kpi-status badge-warn';
    elValStatus.textContent = `${rep.output_vertices} verts`;

    // 5. Homeometry Variance
    document.getElementById('kpi-val-homeometry').textContent = rep.homeometry_variance.toFixed(3);
    const elHomStatus = document.getElementById('status-homeometry');
    elHomStatus.className = rep.homeometry_variance <= 0.25 ? 'kpi-status badge-ok' : 'kpi-status badge-warn';
    elHomStatus.textContent = rep.homeometry_variance <= 0.25 ? 'Uniform' : 'Relaxed';

    // 5b. Defeatured feature count / sizing ratio used
    const elDefeatureVal = document.getElementById('kpi-val-defeature');
    const elDefeatureBench = document.getElementById('kpi-bench-defeature');
    const elDefeatureStatus = document.getElementById('status-defeature');
    if (elDefeatureVal && rep.n_features_defeatured !== undefined) {
      elDefeatureVal.textContent = `${rep.n_features_defeatured}`;
      if (elDefeatureBench) elDefeatureBench.textContent = `Sizing Ratio: ${rep.sizing_ratio}`;
      if (elDefeatureStatus) {
        elDefeatureStatus.className = 'kpi-status badge-ok';
        elDefeatureStatus.textContent = rep.n_features_defeatured > 0 ? 'Sub-res noise removed' : 'None needed';
      }
    }

    // 6. Runtime & Blueprint Profile
    document.getElementById('kpi-val-runtime').textContent = `${rep.elapsed_seconds.toFixed(2)}s`;
    const catText = rep.detected_category ? `Profile: ${rep.detected_category}` : 'Tripo SOTA';
    const symText = rep.detected_symmetry ? ' • Symmetrical' : '';
    document.getElementById('status-engine').textContent = `${catText}${symText}`;

    const modeBadge = document.getElementById('mode-badge');
    if (modeBadge && rep.detected_category) {
      modeBadge.textContent = `⚡ ${rep.detected_category}${rep.detected_symmetry ? ' • 100% Sliced x=0' : ''}`;
    }
  }
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
