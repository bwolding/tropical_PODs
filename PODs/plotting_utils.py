### Plot two variable binned  composites - LOG Y scale

def plot_two_variable_binned_composites_log_y_scale(two_variable_binned_coevolution_composites, color_shading_var, color_shading_var_number_of_samples, min_number_of_obs, color_shading_levels, color_shading_map, colorbar_extend_string, colorbar_tick_levels, colorbar_label_string, scientific_colorbar_boolean, plot_vectors_boolean, leading_lagging_centered_string='centered', save_fig_boolean=False, figure_path_and_name='untitled.png'):
    
    bin_number_of_samples_centered = two_variable_binned_coevolution_composites['bin_number_of_samples_centered']
    bin_mean_delta_BV1_centered = two_variable_binned_coevolution_composites['bin_mean_delta_BV1_centered']
    bin_mean_delta_BV2_centered = two_variable_binned_coevolution_composites['bin_mean_delta_BV2_centered']
    
    bin_number_of_samples_leading = two_variable_binned_coevolution_composites['bin_number_of_samples_leading']
    bin_mean_delta_BV1_leading = two_variable_binned_coevolution_composites['bin_mean_delta_BV1_leading']
    bin_mean_delta_BV2_leading = two_variable_binned_coevolution_composites['bin_mean_delta_BV2_leading']
    
    bin_number_of_samples_lagging = two_variable_binned_coevolution_composites['bin_number_of_samples_lagging']
    bin_mean_delta_BV1_lagging = two_variable_binned_coevolution_composites['bin_mean_delta_BV1_lagging']
    bin_mean_delta_BV2_lagging = two_variable_binned_coevolution_composites['bin_mean_delta_BV2_lagging']
    
    # Create mask for regions with insufficient obs #

    insufficient_obs_mask_colors = color_shading_var_number_of_samples < min_number_of_obs

    # Create "centered" figure #

    fig = plt.figure(figsize=(10, 10))
    plt.yscale('log')

    # Ask for, out of a 1x1 grid, the first axes #

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Column Saturation Fraction', fontdict={'size':24,'weight':'bold'})
    ax.set_ylabel('Precipitation Rate [mm day$^{-1}$]', fontdict={'size':24,'weight':'bold'})
    ax.set(xlim=(0.3, bin_number_of_samples_centered.BV1_bin_midpoint.max()), ylim=(10**-3, 10**2))

    # Axis Ticks #

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=90, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18) 
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #

    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples_centered.BV1_bin_midpoint, bin_number_of_samples_centered.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples_centered.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples_centered.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf #

    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA, color_shading_var.where(~insufficient_obs_mask_colors), levels=color_shading_levels,cmap=color_shading_map, vmin=color_shading_levels.min(), vmax=color_shading_levels.max(), extend=colorbar_extend_string)

    # Speckle regions with insufficient observations #

    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask_colors), BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask_colors), 'ko', ms=1);

    # Quiver the bin mean tendency

    if plot_vectors_boolean:
        
        if leading_lagging_centered_string == 'centered':
            
            vector_too_long_index = (BV2_bin_midpoint_meshgrid_DA + bin_mean_delta_BV2_centered) < (10**-3)
    
            vector_off_plot_index = (BV2_bin_midpoint_meshgrid_DA) < (10**-3)
            
            insufficient_obs_mask_vectors = bin_number_of_samples_centered < min_number_of_obs
            
            plot_vectors_index = ~insufficient_obs_mask_vectors.values & (~vector_too_long_index).values & (~vector_off_plot_index).values
            
            q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2,:], BV2_bin_midpoint_meshgrid_DA[::2,:],\
                          bin_mean_delta_BV1_centered.where(plot_vectors_index)[::2,:], bin_mean_delta_BV2_centered.where(plot_vectors_index)[::2,:], width=0.007,\
                          angles='xy', scale_units='xy', scale=1, pivot='tail') # Very important to have "angles" and "scale_units" set to "xy". "pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
            
        elif leading_lagging_centered_string == 'leading':
            
            LOG_Y_QUIVER_SCALING_FACTOR = 10**(xr.ufuncs.log10(BV2_bin_midpoint_meshgrid_DA) - xr.ufuncs.log10((BV2_bin_midpoint_meshgrid_DA - bin_mean_delta_BV2_leading))) # Only needed when using pivot='tip' with log Y scale
                
            vector_off_plot_index = (BV2_bin_midpoint_meshgrid_DA) < (10**-3)
            
            insufficient_obs_mask_vectors = bin_number_of_samples_leading < min_number_of_obs
            
            plot_vectors_index = ~insufficient_obs_mask_vectors.values & (~vector_off_plot_index).values
            
            q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2,:], BV2_bin_midpoint_meshgrid_DA[::2,:],\
                          bin_mean_delta_BV1_leading.where(plot_vectors_index)[::2,:], (bin_mean_delta_BV2_leading*LOG_Y_QUIVER_SCALING_FACTOR).where(plot_vectors_index)[::2,:], width=0.007,\
                          angles='xy', scale_units='xy', scale=1, pivot='tip') # Very important to have "angles" and "scale_units" set to "xy". # LOG_Y_QUIVER_SCALING_FACTOR only needed when using pivot='tip' with log Y scale"pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
            
        elif leading_lagging_centered_string == 'lagging':
            
            vector_too_long_index = (BV2_bin_midpoint_meshgrid_DA + bin_mean_delta_BV2_lagging) < (10**-3)
    
            vector_off_plot_index = (BV2_bin_midpoint_meshgrid_DA) < (10**-3)
            
            insufficient_obs_mask_vectors = bin_number_of_samples_lagging < min_number_of_obs
            
            plot_vectors_index = ~insufficient_obs_mask_vectors.values & (~vector_too_long_index).values & (~vector_off_plot_index).values
            
            q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2,:], BV2_bin_midpoint_meshgrid_DA[::2,:],\
                          bin_mean_delta_BV1_lagging.where(plot_vectors_index)[::2,:], bin_mean_delta_BV2_lagging.where(plot_vectors_index)[::2,:], width=0.007,\
                          angles='xy', scale_units='xy', scale=1, pivot='tail') # Very important to have "angles" and "scale_units" set to "xy". "pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
            
        else:
            
            print('No plotting convention given, not vectors will be plotted')
        
        #ax.quiverkey(q, X=0, Y=0, U=10, label='Quiver key, length = 1', labelpos='E')
        
        #ax.quiverkey(q, X=0, Y=0, U=10, label='Quiver key, length = 1', labelpos='E')

    # Colorbar # 
    
    if scientific_colorbar_boolean:
        # Colorbar # 
        fmt = tkr.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
    
        cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14, format=fmt)
        cbar.set_ticks(colorbar_tick_levels)
        cbar.ax.get_yaxis().labelpad = 0
        cbar.set_label(colorbar_label_string, rotation=0, fontdict={'size':18,'weight':'bold'})
        for tick in cbar.ax.xaxis.get_majorticklabels():
            tick.set_fontsize(18) 
            tick.set_fontweight('bold')
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(length=10,direction='in')
    
        cbar.ax.xaxis.offsetText.set_fontsize(22)
        cbar.ax.xaxis.offsetText.set_fontweight('bold')
        
    else:
        cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.125)
        cbar.set_ticks(colorbar_tick_levels)
        cbar.ax.get_yaxis().labelpad = 0
        cbar.set_label(colorbar_label_string, rotation=0, fontdict={'size':24,'weight':'bold'})
        for tick in cbar.ax.xaxis.get_majorticklabels():
            tick.set_fontsize(18) 
            tick.set_fontweight('bold')
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(length=10,direction='in')

    # Save figure #
    
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0,
                    frameon=None, metadata=None)