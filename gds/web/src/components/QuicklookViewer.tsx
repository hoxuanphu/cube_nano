import { useEffect, useRef } from "react";
import Map from "ol/Map";
import View from "ol/View";
import Feature from "ol/Feature";
import ImageTile from "ol/ImageTile";
import Projection from "ol/proj/Projection";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import XYZ from "ol/source/XYZ";
import VectorSource from "ol/source/Vector";
import Polygon from "ol/geom/Polygon";
import LineString from "ol/geom/LineString";
import { createBox } from "ol/interaction/Draw";
import Draw from "ol/interaction/Draw";
import Modify from "ol/interaction/Modify";
import Translate from "ol/interaction/Translate";
import Style from "ol/style/Style";
import Fill from "ol/style/Fill";
import Stroke from "ol/style/Stroke";
import type { Geometry } from "ol/geom";
import type { GDSApiClient } from "../api/client";
import type { ProductRef, ROI, Scene } from "../types";
import { clampRoi, roiFromDrag, type SceneBounds } from "../utils/roi";
import { TileCache } from "../utils/tileCache";

type ViewerMode = "pan" | "select";

interface QuicklookViewerProps {
  api: GDSApiClient;
  scene: Scene | null;
  productRef: ProductRef | null;
  tilesEnabled?: boolean;
  showMask?: boolean;
  demoMask?: boolean;
  roi: ROI | null;
  mode: ViewerMode;
  onModeChange: (mode: ViewerMode) => void;
  onRoiChange: (roi: ROI) => void;
  onResetRoi: () => void;
}

const EMPTY_TILE = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

function mapPoint(sceneX: number, sceneY: number, height: number): [number, number] {
  return [sceneX, height - sceneY];
}

function scenePolygon(roi: ROI, height: number): [number, number][][] {
  const [x0, y0] = mapPoint(roi.x, roi.y, height);
  const [x1, y1] = mapPoint(roi.x + roi.width, roi.y + roi.height, height);
  return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]];
}

function sceneRoiFromExtent(extent: [number, number, number, number], bounds: SceneBounds): ROI {
  const [minX, minY, maxX, maxY] = extent;
  return roiFromDrag(
    { x: minX, y: bounds.height - maxY },
    { x: maxX, y: bounds.height - minY },
    bounds,
  );
}

function makeBackdrop(scene: Scene): VectorSource<Feature<Geometry>> {
  const [height, width] = scene.shape;
  const source = new VectorSource<Feature<Geometry>>();
  const frame = new Feature(new Polygon([[[0, 0], [width, 0], [width, height], [0, height], [0, 0]]]));
  frame.setStyle(new Style({ fill: new Fill({ color: "#10242b" }), stroke: new Stroke({ color: "#2e5960", width: 1.5 }) }));
  source.addFeature(frame);
  const step = Math.max(512, Math.round(Math.min(width, height) / 12));
  const gridStyle = new Style({ stroke: new Stroke({ color: "rgba(73, 128, 130, 0.25)", width: 1 }) });
  for (let x = step; x < width; x += step) {
    const line = new Feature(new LineString([[x, 0], [x, height]]));
    line.setStyle(gridStyle);
    source.addFeature(line);
  }
  for (let y = step; y < height; y += step) {
    const line = new Feature(new LineString([[0, y], [width, y]]));
    line.setStyle(gridStyle);
    source.addFeature(line);
  }
  const footprintColors = ["rgba(42, 92, 91, 0.42)", "rgba(47, 78, 105, 0.38)", "rgba(111, 79, 44, 0.30)"];
  const footprintRows = 4;
  const footprintColumns = 5;
  const footprintWidth = width / footprintColumns;
  const footprintHeight = height / footprintRows;
  for (let row = 0; row < footprintRows; row += 1) {
    for (let column = 0; column < footprintColumns; column += 1) {
      const inset = Math.min(18, footprintWidth * 0.02);
      const x0 = column * footprintWidth + inset;
      const y0 = row * footprintHeight + inset;
      const x1 = (column + 1) * footprintWidth - inset;
      const y1 = (row + 1) * footprintHeight - inset;
      const footprint = new Feature(new Polygon([[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]));
      footprint.setStyle(new Style({
        fill: new Fill({ color: footprintColors[(row * footprintColumns + column) % footprintColors.length] }),
        stroke: new Stroke({ color: "rgba(117, 178, 173, 0.35)", width: 1 }),
      }));
      source.addFeature(footprint);
    }
  }
  return source;
}

function makeDemoMaskOverlay(scene: Scene): VectorSource<Feature<Geometry>> {
  const [height, width] = scene.shape;
  const source = new VectorSource<Feature<Geometry>>();
  const zones = [
    { x: 0.22, y: 0.29, w: 0.14, h: 0.08, color: "rgba(239, 119, 116, 0.30)" },
    { x: 0.56, y: 0.58, w: 0.17, h: 0.11, color: "rgba(239, 119, 116, 0.30)" },
    { x: 0.70, y: 0.21, w: 0.08, h: 0.16, color: "rgba(233, 163, 93, 0.28)" },
  ];
  for (const zone of zones) {
    const x0 = zone.x * width;
    const y0 = zone.y * height;
    const x1 = (zone.x + zone.w) * width;
    const y1 = (zone.y + zone.h) * height;
    const feature = new Feature(new Polygon([[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]));
    feature.setStyle(new Style({ fill: new Fill({ color: zone.color }), stroke: new Stroke({ color: "rgba(239, 119, 116, 0.8)", width: 1 }) }));
    source.addFeature(feature);
  }
  return source;
}

export function QuicklookViewer({
  api,
  scene,
  productRef,
  tilesEnabled = true,
  showMask = false,
  demoMask = false,
  roi,
  mode,
  onRoiChange,
  onResetRoi,
}: QuicklookViewerProps) {
  const targetRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const roiSourceRef = useRef<VectorSource<Feature<Geometry>> | null>(null);
  const roiFeatureRef = useRef<Feature<Geometry> | null>(null);
  const interactionsRef = useRef<{ draw: Draw; modify: Modify; translate: Translate } | null>(null);
  const suppressEventsRef = useRef(false);
  const sceneRef = useRef<Scene | null>(null);

  useEffect(() => {
    if (!targetRef.current || !scene) return undefined;
    const [height, width] = scene.shape;
    sceneRef.current = scene;
    const projection = new Projection({
      code: `scene:${scene.scene_ref.catalog_epoch}:${scene.scene_ref.scene_id}:${scene.scene_ref.scene_revision}`,
      units: "pixels",
      extent: [0, 0, width, height],
    });
    const tileCache = new TileCache({ maxEntries: 128, maxBytes: 32 * 1024 * 1024, maxConcurrent: 8 });
    const layers: Array<TileLayer<XYZ> | VectorLayer<VectorSource<Feature<Geometry>>>> = [
      new VectorLayer({ source: makeBackdrop(scene) }),
    ];
    if (productRef && tilesEnabled) {
      const tiles = new TileLayer({
        opacity: 0.82,
        source: new XYZ({
          projection,
          tileSize: 256,
          url: api.tileUrl(productRef, 0, 0, 0).replace(/\/0\/0\/0$/, "/{z}/{x}/{y}"),
          crossOrigin: "anonymous",
          tileLoadFunction: (tile, src) => {
            const image = (tile as ImageTile).getImage() as HTMLImageElement;
            void tileCache.load(src).then((url) => {
              image.src = url;
            }).catch(() => {
              image.src = EMPTY_TILE;
            });
          },
        }),
      });
      layers.push(tiles);
    }
    if (showMask && demoMask) {
      const maskLayer = new VectorLayer({ source: makeDemoMaskOverlay(scene), zIndex: 5 });
      layers.push(maskLayer);
    }
    const roiSource = new VectorSource<Feature<Geometry>>();
    const roiFeature = new Feature<Geometry>();
    roiFeature.setStyle(new Style({
      fill: new Fill({ color: "rgba(229, 154, 65, 0.12)" }),
      stroke: new Stroke({ color: "#e9a35d", width: 3 }),
    }));
    if (roi) roiFeature.setGeometry(new Polygon(scenePolygon(roi, height)));
    roiSource.addFeature(roiFeature);
    const roiLayer = new VectorLayer({ source: roiSource, zIndex: 10 });
    const map = new Map({
      target: targetRef.current,
      layers: [...layers, roiLayer],
      view: new View({ projection, center: [width / 2, height / 2], zoom: 0, maxZoom: 8 }),
      controls: [],
      interactions: [],
    });
    map.getView().fit([0, 0, width, height], { padding: [24, 24, 24, 24], duration: 0 });
    mapRef.current = map;
    roiSourceRef.current = roiSource;
    roiFeatureRef.current = roiFeature;

    const bounds: SceneBounds = { width, height, minPatchSize: 256 };
    const draw = new Draw({ source: roiSource, type: "Circle", geometryFunction: createBox() });
    draw.on("drawend", (event) => {
      roiSource.clear();
      const feature = event.feature as Feature<Geometry>;
      feature.setStyle(roiFeature.getStyle());
      roiSource.addFeature(feature);
      roiFeatureRef.current = feature;
      const geometry = feature.getGeometry();
      if (geometry) onRoiChange(sceneRoiFromExtent(geometry.getExtent() as [number, number, number, number], bounds));
    });
    const modify = new Modify({ source: roiSource });
    modify.on("modifyend", (event) => {
      const feature = event.features.item(0);
      const geometry = feature?.getGeometry();
      if (geometry) onRoiChange(sceneRoiFromExtent(geometry.getExtent() as [number, number, number, number], bounds));
    });
    const translate = new Translate({ features: roiSource.getFeaturesCollection() ?? undefined });
    translate.on("translateend", (event) => {
      const feature = event.features.item(0);
      const geometry = feature?.getGeometry();
      if (geometry) onRoiChange(sceneRoiFromExtent(geometry.getExtent() as [number, number, number, number], bounds));
    });
    interactionsRef.current = { draw, modify, translate };
    if (mode === "select") {
      map.addInteraction(draw);
      map.addInteraction(modify);
      map.addInteraction(translate);
    }
    return () => {
      map.setTarget(undefined);
      tileCache.clear();
      mapRef.current = null;
      roiSourceRef.current = null;
      roiFeatureRef.current = null;
      interactionsRef.current = null;
      sceneRef.current = null;
    };
  }, [api, scene, productRef, tilesEnabled, showMask, demoMask]);

  useEffect(() => {
    const map = mapRef.current;
    const interactions = interactionsRef.current;
    if (!map || !interactions) return;
    for (const interaction of Object.values(interactions)) map.removeInteraction(interaction);
    if (mode === "select") {
      map.addInteraction(interactions.draw);
      map.addInteraction(interactions.modify);
      map.addInteraction(interactions.translate);
    }
  }, [mode]);

  useEffect(() => {
    const feature = roiFeatureRef.current;
    const scene = sceneRef.current;
    if (!feature || !scene || !roi) return;
    suppressEventsRef.current = true;
    feature.setGeometry(new Polygon(scenePolygon(clampRoi(roi, { width: scene.shape[1], height: scene.shape[0], minPatchSize: 256 }), scene.shape[0])));
    suppressEventsRef.current = false;
  }, [roi]);

  return (
    <div className="viewer-wrap">
      <div ref={targetRef} className="quicklook-map" role="img" aria-label={scene ? `Quicklook for scene ${scene.scene_ref.scene_id}` : "No scene selected"} />
      <div className="map-overlay map-overlay-top">
        <span className="map-chip"><span className="live-dot" /> GDS TILE CACHE</span>
        <span className="map-chip">256 PX TILES</span>
        {productRef && <span className="map-chip">PREVIEW {productRef.product_id}</span>}
      </div>
      <div className="map-overlay map-overlay-bottom">
        <span className="map-legend"><span className="legend-swatch scene-swatch" /> quicklook</span>
        <span className="map-legend"><span className="legend-swatch roi-swatch" /> ROI window</span>
        <span className="map-scale">scene pixels / north-up display</span>
      </div>
    </div>
  );
}
