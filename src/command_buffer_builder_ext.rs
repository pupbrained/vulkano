use std::sync::Arc;

use egui_winit_vulkano::Gui;
use vulkano::{command_buffer::{AutoCommandBufferBuilder, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo}, descriptor_set::DescriptorSet, pipeline::{Pipeline, PipelineBindPoint}};

use crate::{RenderContext, VikingRoomModelBuffers};

pub(crate) trait AutoCommandBufferBuilderExt<L> {
    fn build_app_render_pass(
        &mut self,
        rcx: &mut RenderContext,
        descriptor_set: &Arc<DescriptorSet>,
        image_index: u32,
        model_buffers: &VikingRoomModelBuffers,
        gui: &mut Option<Gui>
    );
}

impl<L> AutoCommandBufferBuilderExt<L> for AutoCommandBufferBuilder<L> {
    fn build_app_render_pass(
        &mut self,
        rcx: &mut RenderContext,
        descriptor_set: &Arc<DescriptorSet>,
        image_index: u32,
        model_buffers: &VikingRoomModelBuffers,
        gui: &mut Option<Gui>
    ) {
        self
            .begin_render_pass(
                RenderPassBeginInfo {
                clear_values: vec![
                    Some([0.0, 0.0, 0.0, 1.0].into()), // msaa_color clear value
                    None,                              // final_color (DontCare)
                    Some(1.0.into()),                  // depth clear value
                ],
                ..RenderPassBeginInfo::framebuffer(rcx.framebuffers[image_index as usize].clone())
                },
                SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
                },
            )
            .unwrap();

        self
            .bind_pipeline_graphics(rcx.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                rcx.pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(
                0,
                (
                model_buffers.positions.clone(),
                model_buffers.normals.clone(),
                model_buffers.tex_coords.clone(),
                ),
            )
            .unwrap()
            .bind_index_buffer(model_buffers.indices.clone())
            .unwrap();

        unsafe { self.draw_indexed(model_buffers.indices.len() as u32, 1, 0, 0, 0) }
            .unwrap();

        // Move to the egui subpass
        self
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo {
                contents: SubpassContents::SecondaryCommandBuffers,
                ..Default::default()
                },
            )
            .unwrap();

        // Draw egui in the second subpass
        if let Some(gui_copy) = gui {
        let cb = gui_copy.draw_on_subpass_image([
            rcx.swapchain.image_extent()[0],
            rcx.swapchain.image_extent()[1],
        ]);
        self.execute_commands(cb).unwrap();
        }

        // End the render pass
        self.end_render_pass(SubpassEndInfo::default()).unwrap();
    }
}
