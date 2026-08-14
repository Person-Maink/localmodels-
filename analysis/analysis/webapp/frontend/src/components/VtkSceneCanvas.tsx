import { useEffect, useRef } from "react";

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkCellArray from "@kitware/vtk.js/Common/Core/CellArray";
import vtkPoints from "@kitware/vtk.js/Common/Core/Points";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";

import type { SceneActor } from "../types";

type VtkSceneCanvasProps = {
  staticActors: SceneActor[];
  actors: SceneActor[];
  handVisibility: Record<string, boolean>;
};

function toRgb(color: string): [number, number, number] {
  const normalized = color.replace("#", "");
  const safe = normalized.length === 3
    ? normalized.split("").map((chunk) => chunk + chunk).join("")
    : normalized.padEnd(6, "0").slice(0, 6);
  const r = Number.parseInt(safe.slice(0, 2), 16) / 255;
  const g = Number.parseInt(safe.slice(2, 4), 16) / 255;
  const b = Number.parseInt(safe.slice(4, 6), 16) / 255;
  return [r, g, b];
}

function visibleByHand(actor: SceneActor, handVisibility: Record<string, boolean>) {
  if (!actor.hand) {
    return true;
  }
  return handVisibility[actor.hand] ?? true;
}

function buildPolyDataActor(actor: SceneActor) {
  const points = vtkPoints.newInstance();
  const polyData = vtkPolyData.newInstance();
  const vtkActorInstance = vtkActor.newInstance();
  const mapper = vtkMapper.newInstance();

  points.setData(Float32Array.from(actor.points.flat()), 3);
  polyData.setPoints(points);

  if (actor.kind === "mesh") {
    const polys = vtkCellArray.newInstance({
      values: Uint32Array.from(actor.faces.flatMap((face) => [3, face[0], face[1], face[2]]))
    });
    polyData.setPolys(polys);
  }

  if (actor.kind === "segments") {
    const segmentPoints: number[] = [];
    const lineCells: number[] = [];
    let offset = 0;
    for (const segment of actor.segments) {
      segmentPoints.push(...segment[0], ...segment[1]);
      lineCells.push(2, offset, offset + 1);
      offset += 2;
    }
    const linePointArray = vtkPoints.newInstance();
    linePointArray.setData(Float32Array.from(segmentPoints), 3);
    polyData.setPoints(linePointArray);
    polyData.setLines(vtkCellArray.newInstance({ values: Uint32Array.from(lineCells) }));
  }

  if (actor.kind === "points") {
    const verts = vtkCellArray.newInstance({
      values: Uint32Array.from(actor.points.flatMap((_, index) => [1, index]))
    });
    polyData.setVerts(verts);
  }

  mapper.setInputData(polyData);
  vtkActorInstance.setMapper(mapper);
  const [r, g, b] = toRgb(actor.color);
  vtkActorInstance.getProperty().setColor(r, g, b);
  vtkActorInstance.getProperty().setOpacity(actor.opacity);

  if (actor.kind === "points") {
    vtkActorInstance.getProperty().setRepresentationToPoints();
    vtkActorInstance.getProperty().setPointSize(Number(actor.meta?.point_radius ?? 8));
  }

  if (actor.kind === "segments") {
    vtkActorInstance.getProperty().setLineWidth(Number(actor.meta?.line_width ?? 2));
  }

  return vtkActorInstance;
}

export function VtkSceneCanvas({ staticActors, actors, handVisibility }: VtkSceneCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const renderWindowRef = useRef<vtkGenericRenderWindow | null>(null);
  const vtkActorsRef = useRef<any[]>([]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const genericRenderWindow = vtkGenericRenderWindow.newInstance();
    genericRenderWindow.setContainer(containerRef.current);
    genericRenderWindow.resize();
    const renderer = genericRenderWindow.getRenderer();
    renderer.setBackground(0.98, 0.97, 0.95);
    renderWindowRef.current = genericRenderWindow;

    return () => {
      genericRenderWindow.delete();
      renderWindowRef.current = null;
    };
  }, []);

  useEffect(() => {
    const genericRenderWindow = renderWindowRef.current;
    if (!genericRenderWindow) {
      return;
    }
    const renderer = genericRenderWindow.getRenderer();
    vtkActorsRef.current.forEach((vtkActorInstance) => renderer.removeActor(vtkActorInstance));
    vtkActorsRef.current = [];

    const sceneActors = [...staticActors, ...actors].filter((actor) => visibleByHand(actor, handVisibility));
    for (const actor of sceneActors) {
      const vtkActorInstance = buildPolyDataActor(actor);
      renderer.addActor(vtkActorInstance);
      vtkActorsRef.current.push(vtkActorInstance);
    }

    renderer.resetCamera();
    genericRenderWindow.getRenderWindow().render();
  }, [staticActors, actors, handVisibility]);

  return <div className="vtk-canvas" ref={containerRef} />;
}
