% define the Fourier coefficients
a0 = 3/2;
an = @(n) (1-(-1).^n)./(n.^2*pi^2);
bn = @(n) ((-1).^n)/(n*pi);

% range of (positive) n over which to plot
n=1:10;

% find the line spectrum
c0 = 0.5*sqrt(a0^2);
cn = 0.5*sqrt(an(n).^2 + bn(n).^2);
N = [-fliplr(n) 0 n];
CN = [fliplr(cn) c0 cn];

% and plot
plotspect(N,CN,'r')


function plotspect(n,cn,col)

    % first plot the line peaks
    plot(n,cn,[col 'o'],'LineWidth',2);
    hold on
    for i=1:length(n)
        if cn(i)~=0
            % and now the lines themselves, if they have non-zero height
            plot([n(i) n(i)],[0 cn(i)],[col '-'],'LineWidth',2);
        end
    end
    hold off
    % simple axis labelling
    xlabel('$n \omega$','Interpreter','latex');
    ylabel('$\frac{1}{2}\sqrt{a_n^2+b_n^2}$','Interpreter','latex');

end
