function [gbestX,gbestfitness,gbesthistory]= GGO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin ,maxiter,Func,FuncId,VisualSwitch)


FEs=0;
MaxFEs=popsize*maxiter;
Fitness=Func;
x=xmin+(xmax-xmin)*unifrnd(0,1,popsize,dimension);

gbestfitness=inf;
for i=1:popsize
    fitness(i)=Fitness(x(i,:)',FuncId);
    FEs=FEs+1;
    if gbestfitness>fitness(i)
        gbestfitness=fitness(i);
        gbestX=x(i,:);
    end
    gbesthistory(FEs)=gbestfitness;
    %fprintf("GGO 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
end


while FEs <= MaxFEs

    [~, ind]=sort(fitness);
    Best_X=x(ind(1),:);
    for i=1:popsize

        Worst_X = x(ind(randi([popsize-4,popsize])),:);
        Better_X=x(ind(randi([2,5])),:);
        random=selectID(popsize,i,2);
        L1=random(1);
        L2=random(2);

        D_value1=(Best_X-Better_X);
        D_value2=(Best_X-Worst_X);
        D_value3=(Better_X-Worst_X);
        D_value4=(x(L1,:)-x(L2,:));

        Distance1=norm(D_value1);
        Distance2=norm(D_value2);
        Distance3=norm(D_value3);
        Distance4=norm(D_value4);


        rate=Distance1+Distance2+Distance3+Distance4;
        LF1=Distance1/rate;
        LF2=Distance2/rate;
        LF3=Distance3/rate;
        LF4=Distance4/rate;

        SF=(fitness(i)/max(fitness))^1.25;
        gradw =SF;

        rate1=LF1*SF;
        rate2=LF2*SF;
        rate3=LF3*SF;
        rate4=LF4*SF;

        newx(i,:)=x(i,:)+(rate1*D_value1+rate2*D_value2+rate3*D_value3+rate4*D_value4);
        newx(i,:)=max(newx(i,:),xmin);
        newx(i,:)=min(newx(i,:),xmax);
        newfitness=Fitness(newx(i,:)',FuncId);
        FEs=FEs+1;
        %梯度指导精英粒子
        if ismember(i, ind(1:10))
            gradient=computeGradient(x(i, :), Func, FuncId);
            gradx=x(i,:);
            gradfit=Fitness(gradx',FuncId);
            T=30;
            for j=1:T
                gradw=gradw*0.8;
                newgradx=gradx - gradw * gradient;
                newgradx=max(newgradx,xmin);
                newgradx=min(newgradx,xmax);

                newgradfit=Fitness(newgradx',FuncId);
                if gradfit>newgradfit
                    gradfit=newgradfit;
                    gradx=newgradx;
                end
            end

            %更新最优
            if newfitness>gradfit
                newfitness=gradfit;
                newx(i,:)=gradx;
            end
        end
        if fitness(i)>newfitness
            fitness(i)=newfitness;
            x(i,:)=newx(i,:);
        else
            if rand<0.001&&ind(i)~=ind(1)
                fitness(i)=newfitness;
                x(i,:)=newx(i,:);
            end
        end

        if gbestfitness>fitness(i)
            gbestfitness=fitness(i);
            gbestX=x(i,:);
        end
        gbesthistory(FEs)=gbestfitness;
        if mod(FEs,MaxFEs/10) == 0 && FEs <= MaxFEs
            fprintf("GGO 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
        end
    end

    if FEs>=MaxFEs
        break;
    end

    for i=1:popsize
        newx(i,:)=x(i,:);
        j=1;
        while j<=dimension
            if rand<0.3
                R=x(ind(randi(5)),:);
                newx(i,j) = x(i,j)+(R(:,j)-x(i,j))*unifrnd(0,1);
                if rand<(0.01+(0.1-0.01)*(1-FEs/MaxFEs))
                    newx(i,j)=xmin+(xmax-xmin)*unifrnd(0,1);
                end
            end
            j=j+1;
        end

        newx(i,:)=max(newx(i,:),xmin);
        newx(i,:)=min(newx(i,:),xmax);

        newfitness=Fitness(newx(i,:)',FuncId);
        FEs=FEs+1;
        if fitness(i)>newfitness
            fitness(i)=newfitness;
            x(i,:)=newx(i,:);
        else
            if rand<0.001&&ind(i)~=ind(1)
                fitness(i)=newfitness;
                x(i,:)=newx(i,:);
            end
        end
        if gbestfitness>fitness(i)
            gbestfitness=fitness(i);
            gbestX=x(i,:);
        end
        gbesthistory(FEs)=gbestfitness;
        if mod(FEs,MaxFEs/10) == 0 && FEs <= MaxFEs
            fprintf("GGO 第%d次评价，最佳适应度 = %e\n",FEs,gbestfitness);
        end
    end
end
%%

if FEs<MaxFEs
    gbesthistory(FEs+1:MaxFEs)=gbestfitness;
else
    if FEs>MaxFEs
        gbesthistory(MaxFEs+1:end)=[];
    end
end

end

%% Compute gradient (simplified gradient approximation)
function grad = computeGradient(x, Func, FuncId)
epsilon = 1e-7;
grad = zeros(size(x));
for j = 1:length(x)
    x1 = x;
    x2 = x;
    x1(j) = x1(j) + epsilon;
    x2(j) = x2(j) - epsilon;
    grad(j) = (Func(x1', FuncId) - Func(x2', FuncId)) / (2 * epsilon);
end
end


%% 高效算法: 在[1,popsize]区间内随机选出不同于个体i的count个整数
function r = selectID(popsize,i,count)
% 函数功能：在[1,popsizze]内随机生成count个不包括i的彼此不重复的整数值

% 1.将popsize个整数随机乱序排列
lists = randperm(popsize);

%2.取前count+1个随机整数
r = lists(1:count+1);

% 3. 删除可能包括i的整数(r中也可能没有i)
r(r==i)=[];

% 4. 取前count个不同于i的数
r=r(1:count);

end


